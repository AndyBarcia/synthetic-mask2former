"""YOLOX proposal and object-sequence losses, following SetCriterion's interface."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import masks_to_boxes
from detectron2.utils.events import get_event_storage

from .mask_ops import DynamicMaskRenderer


def mask_losses(logits, targets):
    if not len(logits):
        return logits.sum() * 0, logits.sum() * 0
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none').mean()
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * targets).sum(-1) + 1) / (probability.sum(-1) + targets.sum(-1) + 1)
    return bce, dice.mean()


class YOLOXCriterion(nn.Module, DynamicMaskRenderer):
    """Compute named, unweighted losses; the meta-architecture applies weight_dict.

    The criterion owns matching and the object decoder, like Mask2Former's
    SetCriterion. The semantic head only computes predictions and embeddings.
    """

    def __init__(self, num_classes, matcher, weight_dict, object_decoder, thing_ids,
                 *, full_mask_loss=False, object_rl_weight=0., object_rl_only=False,
                 object_rl_num_samples=4, object_rl_max_steps=64, rendering='forward'):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.object_decoder = object_decoder
        self.full_mask_loss = full_mask_loss
        self.dynamic_masks = True
        self.object_rl_weight = object_rl_weight
        self.object_rl_only = object_rl_only
        self.object_rl_num_samples = object_rl_num_samples
        self.object_rl_max_steps = object_rl_max_steps
        self.rendering = rendering
        thing_ids = sorted(thing_ids)
        self.register_buffer('thing_ids', torch.tensor(thing_ids, dtype=torch.long))
        self.register_buffer('stuff_ids', torch.tensor(sorted(set(range(num_classes)) - set(thing_ids)), dtype=torch.long))
        self.register_buffer('isthing', torch.tensor([i in thing_ids for i in range(num_classes)]))

    def loss_labels(self, raw, target, index, ids, matched, quality):
        norm = max(len(ids), 1)
        objectness = torch.zeros_like(raw['obj'][index])
        objectness[ids] = 1
        classes = torch.zeros_like(raw['cls'][index, ids])
        things = self.matcher.thing_lookup[target['labels']] >= 0
        local_labels = self.matcher.thing_lookup[target['labels'][things]][matched]
        classes[torch.arange(len(ids), device=ids.device), local_labels] = quality.detach()
        loss_cls = F.binary_cross_entropy_with_logits(raw['cls'][index, ids], classes, reduction='sum') / norm
        if self.matcher.soft_labels:
            all_targets = torch.zeros_like(raw['cls'][index])
            all_targets[ids] = classes
            logits = raw['cls'][index]
            loss_cls = (F.binary_cross_entropy_with_logits(logits, all_targets, reduction='none')
                        * (all_targets - logits.sigmoid()).abs().square()).sum() / norm
        return {
            'loss_objectness': F.binary_cross_entropy_with_logits(raw['obj'][index], objectness, reduction='sum') / norm,
            'loss_ce': loss_cls,
        }

    def loss_masks(self, raw, dense, target, index, ids, matched, points):
        things = self.matcher.thing_lookup[target['labels']] >= 0
        logits = self.render_candidates(raw, dense, index, ids, points)
        bce, dice = mask_losses(logits, target['masks'][things][matched].flatten(1)[:, points])
        return {'loss_mask': bce, 'loss_dice': dice}

    def loss_stuff(self, raw, dense, target, index, points):
        masks = dense.new_zeros((len(self.stuff_ids), *dense.shape[-2:]))
        for k, category in enumerate(self.stuff_ids):
            rows = target['masks'][target['labels'] == category]
            if len(rows):
                masks[k] = rows.amax(0)
        present = masks.flatten(1).any(-1)
        logits = self.render(raw['stuff_kernels'][index, present], dense[index])[:, points]
        bce, dice = mask_losses(logits, masks[present].flatten(1)[:, points])
        return {
            'loss_stuff_presence': F.binary_cross_entropy_with_logits(raw['stuff_presence'][index], present.float()),
            'loss_stuff_mask': bce, 'loss_stuff_dice': dice,
        }

    def loss_proposals(self, outputs, targets):
        raw, dense = outputs['dense_predictions'], outputs['mask_features']
        prepared = []
        for target in targets:
            valid = target['masks'].flatten(1).any(-1)
            full_masks = target['masks'][valid]
            prepared.append(dict(labels=target['labels'][valid], full_masks=full_masks,
                                 masks=F.interpolate(full_masks[:, None].float(), dense.shape[-2:], mode='nearest')[:, 0]))
        indices = self.matcher(outputs, prepared)
        losses = {}
        for index, (target, (ids, matched)) in enumerate(zip(prepared, indices)):
            things = self.matcher.thing_lookup[target['labels']] >= 0
            image_losses = {}
            if self.matcher.box_free:
                with torch.no_grad():
                    binary = self.render_candidates(raw, dense, index, ids) > 0
                    gt = target['masks'][things][matched].flatten(1)
                    intersection = (binary * gt).sum(-1)
                    quality = (intersection / (binary.sum(-1) + gt.sum(-1) - intersection).clamp_min(1)).sqrt().clamp_min(.1)
            else:
                predicted = raw['boxes'][index, ids]
                boxes = masks_to_boxes(target['full_masks'][things].bool())
                if len(boxes):
                    boxes[:, 2:] += 1
                boxes = boxes[matched]
                intersection = (torch.minimum(predicted[:, 2:], boxes[:, 2:])
                                - torch.maximum(predicted[:, :2], boxes[:, :2])).clamp_min(0).prod(-1)
                union = ((predicted[:, 2:] - predicted[:, :2]).clamp_min(0).prod(-1)
                         + (boxes[:, 2:] - boxes[:, :2]).prod(-1) - intersection)
                quality = intersection / union.clamp_min(1e-8)
                image_losses['loss_box'] = (1 - quality.square()).mean() if len(ids) else raw['boxes'][index].sum() * 0
            image_losses.update(self.loss_labels(raw, target, index, ids, matched, quality))
            points = (torch.arange(dense.shape[-2] * dense.shape[-1], device=dense.device) if self.full_mask_loss
                      else torch.randperm(dense.shape[-2] * dense.shape[-1], device=dense.device)[:2048])
            image_losses.update(self.loss_masks(raw, dense, target, index, ids, matched, points))
            image_losses.update(self.loss_stuff(raw, dense, target, index, points))
            for name, loss in image_losses.items():
                losses[name] = losses.get(name, 0.) + loss / len(targets)
        return losses

    def loss_object_decoder(self, outputs, targets):
        decoder = self.object_decoder
        embeddings, masks = outputs['mask_embeddings'], outputs['pred_masks']
        matches = self.matcher.match_objects(outputs['proposals'], targets)
        paths = [[matched[torch.randperm(len(matched), device=masks.device)].tolist()
                  for _ in range(decoder.prefix_tree_branches)] for matched in matches]
        previous, positions, ancestors, weights = decoder.pack_prefix_trees(paths, decoder.num_queries, masks.device)
        regions = decoder.prepare_regions(masks, outputs['object_decoder_image_sizes'])
        logits = decoder(embeddings, outputs['object_decoder_image_features'], regions, previous,
                         decoder.prepare_vocabulary_exclusions(masks), positions=positions, ancestors=ancestors)
        return {'loss_object_decoder': -(logits.float().log_softmax(-1) * weights).sum() / weights.sum()}

    def loss_object_rl(self, outputs, targets):
        from .rl import prepare_pq_reward, score_orders, trajectory_log_probability, rloo_loss
        decoder = self.object_decoder
        embeddings, masks = outputs['mask_embeddings'], outputs['pred_masks']
        memories, sizes = outputs['object_decoder_image_features'], outputs['object_decoder_image_sizes']
        samples, batch = self.object_rl_num_samples, len(targets)
        with torch.no_grad():
            repeated_memories = [x.repeat(samples, 1, 1) for x in memories]
            repeated_masks = masks.repeat(samples, 1, 1, 1)
            orders, _, diagnostics, actions = decoder.rollout(
                embeddings.repeat(samples, 1, 1), repeated_memories, repeated_masks, sizes, sample=True,
                max_steps=self.object_rl_max_steps, return_diagnostics=True, return_actions=True)
            greedy, _ = decoder.rollout(embeddings, memories, masks, sizes, sample=False, max_steps=self.object_rl_max_steps)
            prepared = prepare_pq_reward(outputs['proposals'], targets, outputs['padded_size'])
            rewards = score_orders(orders, prepared, self.isthing, self.num_classes, self.rendering).reshape(samples, batch)
            greedy_rewards = score_orders(greedy, prepared, self.isthing, self.num_classes, self.rendering)
        logp = trajectory_log_probability(decoder, embeddings.repeat(samples, 1, 1), repeated_memories,
                                         repeated_masks, sizes, actions).reshape(samples, batch)
        loss, advantage = rloo_loss(rewards, greedy_rewards, logp)
        try:
            get_event_storage().put_scalars(**{'rl/sampled_pq': rewards.mean(), 'rl/greedy_pq': greedy_rewards.mean(),
                                              'rl/advantage_abs': advantage.abs().mean()})
        except AssertionError:
            pass
        return {'loss_object_rl': loss}

    def forward(self, outputs, targets):
        if self.object_rl_only:
            return self.loss_object_rl(outputs, targets)
        losses = self.loss_proposals(outputs, targets)
        losses.update(self.loss_object_decoder(outputs, targets))
        if self.object_rl_weight > 0:
            losses.update(self.loss_object_rl(outputs, targets))
        return losses
