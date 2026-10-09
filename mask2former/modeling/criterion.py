# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from ..utils.rl_logging import record_best_of_n_diagnostics, record_rl_diagnostics
from .utils import compute_mask_block_counts


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def rloo_self_critical_loss(sampled_rewards, greedy_rewards, log_probabilities):
    """Leave-one-out policy gradient, with greedy reward as one baseline member.

    Tensors have shapes [rollouts, images], [images], and [rollouts, images].
    """
    if sampled_rewards.shape != log_probabilities.shape or sampled_rewards.ndim != 2:
        raise ValueError("sampled rewards and log probabilities must have shape [K, batch]")
    if greedy_rewards.shape != sampled_rewards.shape[1:]:
        raise ValueError("greedy rewards must have shape [batch]")
    k = sampled_rewards.shape[0]
    if k < 1:
        raise ValueError("at least one sampled rollout is required")
    baseline = (greedy_rewards.unsqueeze(0) + sampled_rewards.sum(0, keepdim=True)
                - sampled_rewards) / k
    advantage = sampled_rewards - baseline.detach()
    return -(advantage * log_probabilities).mean(), advantage


def scst_loss(sampled_rewards, greedy_rewards, log_probabilities):
    """Use the greedy reward as the baseline for every sampled rollout."""
    if sampled_rewards.shape != log_probabilities.shape or sampled_rewards.ndim != 2:
        raise ValueError("sampled rewards and log probabilities must have shape [K, batch]")
    if greedy_rewards.shape != sampled_rewards.shape[1:]:
        raise ValueError("greedy rewards must have shape [batch]")
    if sampled_rewards.shape[0] < 1:
        raise ValueError("at least one sampled rollout is required")
    advantage = sampled_rewards - greedy_rewards.detach().unsqueeze(0)
    return -(advantage * log_probabilities).mean(), advantage


def select_best_of_n(sampled_rewards, greedy_rewards, trajectories):
    """Select one sampled trajectory per image and mark strict greedy wins."""
    if sampled_rewards.ndim != 2 or greedy_rewards.shape != sampled_rewards.shape[1:]:
        raise ValueError("rewards must have shapes [K, batch] and [batch]")
    k, batch_size = sampled_rewards.shape
    if len(trajectories) != k * batch_size:
        raise ValueError("trajectories must contain K batches in sample-major order")
    best_rewards, best_indices = sampled_rewards.max(0)
    best_trajectories = [
        trajectories[index * batch_size + image]
        for image, index in enumerate(best_indices.tolist())
    ]
    return best_trajectories, best_rewards, best_rewards > greedy_rewards


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 mask_loss_type="point", object_decoder=None, object_rl_weight=0.0,
                 object_rl_max_steps=101, object_rl_reward_size=0,
                 object_rl_num_samples=4, object_rl_train_eof=True,
                 object_rl_baseline="rloo_greedy", object_rl_objective="policy_gradient",
                 fused_thing_masks=False, thing_class_ids=()):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative weight applied to unmatched queries in the query-bias loss
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.fused_thing_masks = fused_thing_masks
        self.thing_class_ids = tuple(thing_class_ids)
        self.object_decoder = object_decoder
        self.object_rl_weight = object_rl_weight
        self.object_rl_max_steps = object_rl_max_steps
        if object_rl_num_samples < 1:
            raise ValueError("OBJECT_RL_NUM_SAMPLES must be at least one")
        self.object_rl_num_samples = object_rl_num_samples
        if object_rl_baseline not in ("rloo_greedy", "scst"):
            raise ValueError("OBJECT_RL_BASELINE must be 'rloo_greedy' or 'scst'")
        self.object_rl_baseline = object_rl_baseline
        if object_rl_objective not in ("policy_gradient", "best_of_n_ft", "best_of_n_set"):
            raise ValueError(
                "OBJECT_RL_OBJECTIVE must be 'policy_gradient', 'best_of_n_ft', "
                "or 'best_of_n_set'"
            )
        self.object_rl_objective = object_rl_objective
        if object_rl_objective in ("best_of_n_ft", "best_of_n_set") and not object_rl_train_eof:
            raise ValueError("Best-of-N fine-tuning requires EOF training")
        self.object_rl_train_eof = object_rl_train_eof
        if object_rl_weight > 0 and object_rl_train_eof:
            if object_decoder is None:
                raise ValueError("EOF training requires an object decoder")
            if object_rl_max_steps < object_decoder.num_queries + 1:
                raise ValueError("EOF training requires OBJECT_RL_MAX_STEPS >= num_queries + 1")
        if object_rl_reward_size < 0:
            raise ValueError("OBJECT_RL_REWARD_SIZE must be nonnegative")
        self.object_rl_reward_size = object_rl_reward_size
        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        if mask_loss_type not in {"point", "block"}:
            raise ValueError(
                f"MASK_LOSS_TYPE must be 'point' or 'block', got {mask_loss_type!r}"
            )
        self.mask_loss_type = mask_loss_type

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classify matched queries among foreground classes only."""
        assert "pred_logits" in outputs
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # Keep the legacy output shape for checkpoint compatibility, but never
        # normalize over or train its final (background) logit.
        src_logits = outputs["pred_logits"][idx][..., :self.num_classes].float()
        if src_logits.shape[0] == 0:
            return {"loss_ce": src_logits.sum()}
        return {"loss_ce": F.cross_entropy(src_logits, target_classes_o)}

    def loss_query_bias(self, outputs, targets, indices, num_masks):
        """Supervise matched GT/query pairs as positives, other pairs as negatives."""
        if "gt_query_bias_logits" in outputs:
            numerator = outputs["pred_logits"].sum() * 0.0
            denominator = numerator.detach().clone()
            for logits, (src_indices, gt_indices) in zip(
                outputs["gt_query_bias_logits"], indices
            ):
                logits = logits.float()
                target = torch.zeros_like(logits)
                target[gt_indices, src_indices] = 1.0
                weights = torch.where(target.bool(), 1.0, self.eos_coef)
                numerator = numerator + (
                    F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                    * weights
                ).sum()
                denominator = denominator + weights.sum()
            return {"loss_query_bias": numerator / denominator.clamp_min(1.0)}

        assert "query_bias_logits" in outputs
        logits = outputs["query_bias_logits"].float()
        target = torch.zeros_like(logits)
        for batch_index, (src_indices, _) in enumerate(indices):
            target[batch_index, src_indices] = 1.0

        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weights = torch.where(target.bool(), 1.0, self.eos_coef)
        return {"loss_query_bias": (loss * weights).sum() / weights.sum().clamp_min(1.0)}

    def loss_object_decoder(self, outputs, indices):
        """Teacher-force shared prefixes and permuted suffixes in one tree pass."""
        embeddings = outputs["mask_embeddings"]
        num_queries = embeddings.shape[1]
        paths_per_image = []
        for src_indices, _ in indices:
            src_indices = src_indices.to(embeddings.device)
            ordered = src_indices[torch.randperm(len(src_indices), device=embeddings.device)]
            # Keep at least two suffix objects when possible, so branching can
            # change the order. Empty and singleton scenes remain valid paths.
            split = int(torch.randint(max(len(ordered) - 1, 1), (), device=embeddings.device))
            prefix, suffix = ordered[:split], ordered[split:]
            paths = [ordered.tolist()]
            for _ in range(self.object_decoder.prefix_tree_branches - 1):
                permutation = torch.randperm(len(suffix), device=embeddings.device)
                paths.append(torch.cat((prefix, suffix[permutation])).tolist())
            paths_per_image.append(paths)
        previous, positions, ancestors, weights = self.object_decoder.pack_prefix_trees(
            paths_per_image, num_queries, embeddings.device
        )
        image_regions = self.object_decoder.prepare_regions(
            outputs["pred_masks"], outputs["object_decoder_image_sizes"]
        )
        vocabulary_exclusions = self.object_decoder.prepare_vocabulary_exclusions(
            outputs["pred_masks"]
        )
        logits = self.object_decoder(
            embeddings, outputs["object_decoder_image_features"], image_regions, previous,
            vocabulary_exclusions, positions=positions, ancestors=ancestors,
        )
        loss = -(F.log_softmax(logits.float(), dim=-1) * weights).sum() / weights.sum()
        return {"loss_object_decoder": loss}

    @torch.no_grad()
    def object_reward(self, orders, outputs, targets):
        """Per-image PQ rewards via the submodule's stateless batched API.

        Training targets share padded image dimensions and are scored together.
        Preserve reward resolution, stuff merging and painting order.
        """
        submodule = str(Path(__file__).resolve().parents[2] / "panoptic-evaluator")
        if submodule not in sys.path:
            sys.path.insert(0, submodule)
        from panoptic_evaluator import PanopticBatch, panoptic_quality

        device = outputs["pred_masks"].device
        if not orders:
            return torch.empty(0, device=device, dtype=torch.float32)
        batch_size, num_queries = outputs["pred_masks"].shape[:2]
        if len(orders) % batch_size:
            raise ValueError("orders must contain whole batches in sample-major order")
        size = ((self.object_rl_reward_size,) * 2 if self.object_rl_reward_size
                else targets[0]["masks"].shape[-2:])
        masks = F.interpolate(outputs["pred_masks"].float(), size=size,
                              mode="bilinear", align_corners=False) > 0
        classes = outputs["pred_logits"][..., :-1].argmax(-1).to(torch.int32)

        # Only ragged target packing needs an image loop. Resize all GT masks once.
        counts = [len(target["labels"]) for target in targets]
        capacity = max(counts) + 1
        gt_labels = torch.full((batch_size, capacity), -1, device=device, dtype=torch.int32)
        gt_maps = torch.zeros((batch_size, *size), device=device, dtype=torch.int32)
        if sum(counts):
            resized = F.interpolate(torch.cat([t["masks"] for t in targets])[:, None].float(),
                                    size=size, mode="nearest")[:, 0] > 0
            for image, target_masks in enumerate(resized.split(counts)):
                count = counts[image]
                gt_labels[image, 1:count + 1] = targets[image]["labels"]
                if count:
                    ids = torch.arange(1, count + 1, device=device, dtype=torch.int32)
                    gt_maps[image] = (target_masks * ids[:, None, None]).amax(0)

        # A query's last occurrence wins, including repeated queries. Segment IDs
        # can be query IDs: PQ ignores unused slots. Merge all stuff into slot Q+1.
        paint_orders = [list(reversed(order)) if self.object_rl_paint_order == "reverse"
                        else list(order) for order in orders]
        steps = max(map(len, paint_orders))
        padded = torch.tensor([order + [num_queries] * (steps - len(order))
                               for order in paint_orders], device=device, dtype=torch.long)
        ranks = torch.zeros((len(orders), num_queries + 1), device=device, dtype=torch.long)
        if steps:
            ranks.scatter_reduce_(1, padded,
                                  torch.arange(1, steps + 1, device=device)[None].expand_as(padded),
                                  reduce="amax", include_self=True)
        pred_labels = torch.cat((torch.full((batch_size, 1), -1, device=device, dtype=torch.int32),
                                 classes.masked_fill(classes == 0, -1),
                                 torch.zeros((batch_size, 1), device=device, dtype=torch.int32)), 1)
        # Bound the temporary [rollouts*images, queries, pixels] rank tensor.
        chunk_size = max(1, 16_000_000 // max(1, num_queries * size[0] * size[1]))
        rewards = []
        for start in range(0, len(orders), chunk_size):
            stop = min(start + chunk_size, len(orders))
            images = torch.arange(start, stop, device=device) % batch_size
            winning_rank, query = (masks[images] * ranks[start:stop, :num_queries, None, None]).max(1)
            categories = classes[images].gather(1, query.flatten(1)).reshape_as(query)
            painted = torch.where(categories == 0, num_queries + 1, query + 1)
            painted = painted.masked_fill(winning_rank == 0, 0).to(torch.int32)
            rewards.append(panoptic_quality(
                PanopticBatch(painted, pred_labels[images]),
                PanopticBatch(gt_maps[images], gt_labels[images]),
                self.num_classes, validate=False,
            ).float())
        return torch.cat(rewards)

    @staticmethod
    def _repeat_policy_inputs(policy_inputs, repeats):
        """Pack rollout copies in sample-major order for decoding and scoring."""
        embeddings, features, masks, image_sizes = policy_inputs
        return (embeddings.repeat(repeats, 1, 1),
                [feature.repeat(repeats, 1, 1) for feature in features],
                masks.repeat(repeats, 1, 1, 1), image_sizes)

    def _object_rollout_rewards(self, sampled, greedy, outputs, targets):
        """Reuse rendering inputs for all samples and the greedy baseline."""
        batch_size = outputs["pred_masks"].shape[0]
        rewards = self.object_reward(sampled + greedy, outputs, targets)
        return rewards[:-batch_size].reshape(-1, batch_size), rewards[-batch_size:]

    def loss_object_rl(self, outputs, targets):
        if self.object_rl_objective in ("best_of_n_ft", "best_of_n_set"):
            return self.loss_object_best_of_n(outputs, targets)
        if self.object_rl_train_eof:
            return self.loss_object_rl_eof(outputs, targets)
        # Detaching all policy inputs confines this loss to the object decoder.
        policy_inputs = (
            outputs["mask_embeddings"].detach(),
            [feature.detach() for feature in outputs["object_decoder_image_features"]],
            outputs["pred_masks"].detach(),
            outputs["object_decoder_image_sizes"],
        )
        batch_size, num_queries = policy_inputs[0].shape[:2]
        proposal_counts = torch.tensor(
            [len(target["labels"]) for target in targets],
            device=policy_inputs[0].device,
        )
        if (proposal_counts > num_queries).any():
            raise ValueError("GT instance count exceeds the number of mask proposals")
        # The GT supplies the proposal count only. Query bias selects which
        # predictions are eligible, without using GT identities or masks.
        proposal_rank = outputs["query_bias_logits"].detach().argsort(
            dim=-1, descending=True, stable=True
        )
        proposal_mask = torch.zeros(
            (batch_size, num_queries), dtype=torch.bool, device=policy_inputs[0].device
        )
        proposal_mask.scatter_(1, proposal_rank, (
            torch.arange(num_queries, device=proposal_counts.device)[None, :] <
            proposal_counts[:, None]
        ))
        k = self.object_rl_num_samples
        rollout_batch = 4
        sampled = []
        with torch.no_grad():
            for start in range(0, k, rollout_batch):
                repeats = min(rollout_batch, k - start)
                sampled_inputs = self._repeat_policy_inputs(policy_inputs, repeats)
                orders, _ = self.object_decoder.rollout_permutation(
                    *sampled_inputs, proposal_mask.repeat(repeats, 1), sample=True
                )
                sampled.extend(orders)
            greedy, _ = self.object_decoder.rollout_permutation(
                *policy_inputs, proposal_mask, sample=False
            )
            sampled_reward, greedy_reward = self._object_rollout_rewards(
                sampled, greedy, outputs, targets
            )
        # Sampling needs no graph. A full causal pass scores each fixed order,
        # avoiding a growing-prefix autograd graph for every sampled action.
        log_probabilities = []
        for start in range(0, k, rollout_batch):
            repeats = min(rollout_batch, k - start)
            scored_inputs = self._repeat_policy_inputs(policy_inputs, repeats)
            scores = self.object_decoder.permutation_log_probability(
                *scored_inputs, proposal_mask.repeat(repeats, 1),
                sampled[start * batch_size:(start + repeats) * batch_size],
            )
            log_probabilities.append(scores.reshape(repeats, batch_size))
        log_probability = torch.cat(log_probabilities)
        rl_loss, advantage = self._rl_policy_loss(
            sampled_reward, greedy_reward, log_probability
        )
        # An all-empty batch still needs a graph-connected zero for RL-only runs.
        rl_loss = rl_loss + self.object_decoder.bos.sum() * 0.0
        record_rl_diagnostics(sampled_reward, greedy_reward, advantage,
                              log_probability, rl_loss)
        return {"loss_object_rl": rl_loss}

    def _rl_policy_loss(self, sampled_reward, greedy_reward, log_probability):
        loss_fn = scst_loss if self.object_rl_baseline == "scst" else rloo_self_critical_loss
        return loss_fn(sampled_reward, greedy_reward, log_probability)

    def loss_object_best_of_n(self, outputs, targets):
        """Imitate the best sample's ordered trajectory or unordered mask set."""
        query_bias = outputs.get("query_bias_logits")
        query_bias = query_bias.detach() if query_bias is not None else None
        policy_inputs = (
            outputs["mask_embeddings"].detach(),
            [feature.detach() for feature in outputs["object_decoder_image_features"]],
            outputs["pred_masks"].detach(),
            outputs["object_decoder_image_sizes"],
        )
        batch_size = policy_inputs[0].shape[0]
        k = self.object_rl_num_samples
        rollout_batch = 4
        sampled_orders = []
        trajectories = []
        with torch.no_grad():
            for start in range(0, k, rollout_batch):
                repeats = min(rollout_batch, k - start)
                sampled_inputs = self._repeat_policy_inputs(policy_inputs, repeats)
                orders, _, actions = self.object_decoder.rollout(
                    *sampled_inputs, sample=True, max_steps=self.object_rl_max_steps,
                    return_actions=True,
                    query_bias_logits=query_bias.repeat(repeats, 1) if query_bias is not None else None,
                )
                sampled_orders.extend(orders)
                trajectories.extend(actions)
            greedy_orders, _, _ = self.object_decoder.rollout(
                *policy_inputs, sample=False, max_steps=self.object_rl_max_steps,
                return_actions=True,
                query_bias_logits=query_bias,
            )
            sampled_reward, greedy_reward = self._object_rollout_rewards(
                sampled_orders, greedy_orders, outputs, targets
            )
            best_trajectories, best_reward, winners = select_best_of_n(
                sampled_reward, greedy_reward, trajectories
            )
        if self.object_rl_objective == "best_of_n_set":
            loss, mask_nll, eof_nll, token_count, mask_count = (
                self.object_decoder.set_imitation_loss(
                    *policy_inputs, [trajectory[:-1] for trajectory in best_trajectories]
                )
            )
            metric_group = "set_imitation"
        else:
            if winners.any():
                winning_trajectories = [
                    trajectory for trajectory, wins in zip(best_trajectories, winners.tolist())
                    if wins
                ]
                log_probability = self.object_decoder.autoregressive_log_probability(
                    policy_inputs[0][winners],
                    [feature[winners] for feature in policy_inputs[1]],
                    policy_inputs[2][winners], policy_inputs[3], winning_trajectories,
                    query_bias_logits=query_bias[winners] if query_bias is not None else None,
                )
                token_count = sum(map(len, winning_trajectories))
                loss = -log_probability.sum() / token_count
            else:
                token_count = 0
                loss = self.object_decoder.eof.sum() * 0.0
            metric_group = "best_of_n"
            mask_nll = eof_nll = mask_count = None
        record_best_of_n_diagnostics(
            sampled_reward, greedy_reward, best_reward, winners,
            greedy_orders, best_trajectories, loss, token_count,
            metric_group=metric_group, mask_nll=mask_nll, eof_nll=eof_nll,
            mask_count=mask_count,
        )
        return {"loss_object_rl": loss}

    def loss_object_rl_eof(self, outputs, targets):
        """RLOO over free decoding, including the sampled EOF action."""
        query_bias = outputs.get("query_bias_logits")
        query_bias = query_bias.detach() if query_bias is not None else None
        policy_inputs = (
            outputs["mask_embeddings"].detach(),
            [feature.detach() for feature in outputs["object_decoder_image_features"]],
            outputs["pred_masks"].detach(),
            outputs["object_decoder_image_sizes"],
        )
        batch_size = policy_inputs[0].shape[0]
        k = self.object_rl_num_samples
        rollout_batch = 4
        sampled = []
        trajectories = []
        with torch.no_grad():
            for start in range(0, k, rollout_batch):
                repeats = min(rollout_batch, k - start)
                sampled_inputs = self._repeat_policy_inputs(policy_inputs, repeats)
                orders, _, actions = self.object_decoder.rollout(
                    *sampled_inputs, sample=True, max_steps=self.object_rl_max_steps,
                    return_actions=True,
                    query_bias_logits=query_bias.repeat(repeats, 1) if query_bias is not None else None,
                )
                sampled.extend(orders)
                trajectories.extend(actions)
            greedy, _, greedy_actions = self.object_decoder.rollout(
                *policy_inputs, sample=False, max_steps=self.object_rl_max_steps,
                return_actions=True,
                query_bias_logits=query_bias,
            )
            sampled_reward, greedy_reward = self._object_rollout_rewards(
                sampled, greedy, outputs, targets
            )
        log_probabilities = []
        for start in range(0, k, rollout_batch):
            repeats = min(rollout_batch, k - start)
            scored_inputs = self._repeat_policy_inputs(policy_inputs, repeats)
            scores = self.object_decoder.autoregressive_log_probability(
                *scored_inputs,
                trajectories[start * batch_size:(start + repeats) * batch_size],
                query_bias_logits=query_bias.repeat(repeats, 1) if query_bias is not None else None,
            )
            log_probabilities.append(scores.reshape(repeats, batch_size))
        log_probability = torch.cat(log_probabilities)
        rl_loss, advantage = self._rl_policy_loss(
            sampled_reward, greedy_reward, log_probability
        )
        rl_loss = rl_loss + self.object_decoder.bos.sum() * 0.0
        sampled_lengths = sampled_reward.new_tensor(
            [len(order) for order in sampled]
        ).reshape(k, batch_size)
        greedy_lengths = greedy_reward.new_tensor([len(order) for order in greedy])
        record_rl_diagnostics(sampled_reward, greedy_reward, advantage,
                              log_probability, rl_loss,
                              sampled_lengths=sampled_lengths,
                              greedy_lengths=greedy_lengths)
        return {"loss_object_rl": rl_loss}

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        if src_masks.shape[0] == 0:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        if self.mask_loss_type == "block":
            logits = src_masks.flatten(1)
            positive_counts, block_area, target_h, target_w = compute_mask_block_counts(
                target_masks, src_masks.shape[-2:]
            )
            positive_counts = positive_counts.to(logits)

            # This is algebraically identical to dense BCE, while retaining one
            # logit and one positive-pixel count per prediction block.
            loss_mask = (
                block_area * F.softplus(logits) - logits * positive_counts
            ).sum(1) / (target_h * target_w)

            probabilities = logits.sigmoid()
            numerator = 2 * (probabilities * positive_counts).sum(1)
            denominator = (
                block_area * probabilities.sum(1) + positive_counts.sum(1)
            )
            loss_dice = 1 - (numerator + 1) / (denominator + 1)
            return {
                "loss_mask": loss_mask.sum() / num_masks,
                "loss_dice": loss_dice.sum() / num_masks,
            }

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'query_bias': self.loss_query_bias,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def _mask_targets(self, targets):
        """Append mask-only unions without changing the object ground truth."""
        if not self.fused_thing_masks:
            return targets
        result = []
        for target in targets:
            labels, masks = target["labels"], target["masks"]
            fused_labels, fused_masks = [], []
            for category in self.thing_class_ids:
                members = labels == category
                if members.sum().item() > 1:
                    fused_labels.append(labels[members][:1])
                    fused_masks.append(masks[members].bool().any(0).to(masks.dtype)[None])
            result.append(dict(
                target,
                labels=torch.cat([labels] + fused_labels),
                masks=torch.cat([masks] + fused_masks),
                num_object_targets=len(labels),
            ))
        return result

    @staticmethod
    def _object_indices(indices, targets):
        return [(src[tgt < len(target["labels"])], tgt[tgt < len(target["labels"])])
                for (src, tgt), target in zip(indices, targets)]

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        mask_targets = self._mask_targets(targets)
        if self.fused_thing_masks and any(
            len(target["labels"]) > outputs["pred_masks"].shape[1]
            for target in mask_targets
        ):
            raise ValueError(
                "Instance and fused mask targets exceed NUM_OBJECT_QUERIES; "
                "increase it to supervise every target."
            )

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, mask_targets)
        object_indices = self._object_indices(indices, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in mask_targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            loss_targets, loss_indices = ((mask_targets, indices) if loss == "masks"
                                          else (targets, object_indices))
            losses.update(self.get_loss(loss, outputs, loss_targets, loss_indices, num_masks))
        if self.object_decoder is not None:
            losses.update(self.loss_object_decoder(outputs, object_indices))
            if self.object_rl_weight > 0:
                losses.update(self.loss_object_rl(outputs, targets))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, mask_targets)
                object_indices = self._object_indices(indices, targets)
                for loss in self.losses:
                    loss_targets, loss_indices = ((mask_targets, indices) if loss == "masks"
                                                  else (targets, object_indices))
                    l_dict = self.get_loss(loss, aux_outputs, loss_targets, loss_indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
            "mask_loss_type: {}".format(self.mask_loss_type),
            "fused_thing_masks: {}".format(self.fused_thing_masks),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
