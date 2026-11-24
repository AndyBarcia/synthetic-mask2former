"""Repeated whole-object matching slots and one-proposal-per-object teachers."""
import torch
from detectron2.utils.comm import get_world_size
from ..utils.misc import is_dist_avail_and_initialized


def repeat_object_targets(targets, copies):
    """Only whole objects enter matching; parts/fused annotations are excluded."""
    return [dict(labels=t['labels'].repeat_interleave(copies),
                 masks=t['masks'].repeat_interleave(copies, dim=0)) for t in targets]


@torch.no_grad()
def filter_extra_proposals(outputs, targets, indices, copies, min_iou,
                           require_correct_class=False, extra_iou_thresholds=()):
    """Filter fixed-k matches, retaining the highest-IoU match for every represented GT.

    Rejected queries become unmatched/no-object. Indices still refer to repeated
    target slots; teachers map them back to original GTs after filtering.
    """
    from .matcher import pairwise_binary_mask_iou
    filtered = []
    for batch_index, (logits, target, (src, dst)) in enumerate(zip(outputs['pred_masks'], targets, indices)):
        if not len(src):
            filtered.append((src, dst))
            continue
        parents = (dst // copies).to(logits.device)
        ious = pairwise_binary_mask_iou(logits[src.to(logits.device)],
                                        target['masks'].to(logits.device))
        quality = ious[torch.arange(len(src), device=logits.device), parents]
        keep = quality > min_iou
        best_positions = []
        for obj in parents.unique():
            positions = (parents == obj).nonzero(as_tuple=True)[0]
            ordered = positions[quality[positions].argsort(descending=True, stable=True)]
            best_positions.append(ordered[0])
            if extra_iou_thresholds:
                keep[positions] = False
                for rank, position in enumerate(ordered[1:]):
                    keep[position] = quality[position] > extra_iou_thresholds[rank]
        if require_correct_class:
            # Include no-object in argmax: a background prediction is incorrect.
            classes = outputs['pred_logits'][batch_index].argmax(-1)
            keep &= classes[src.to(logits.device)] == target['labels'].to(logits.device)[parents]
        keep[torch.stack(best_positions)] = True
        filtered.append((src[keep.to(src.device)], dst[keep.to(dst.device)]))
    return filtered


def generate_proposal_permutations(indices, k, device):
    """Sample-major candidates: one uniformly sampled query per represented GT."""
    if k < 1:
        raise ValueError('At least one teacher candidate is required')
    groups = [[src[dst == parent].to(device) for parent in dst.unique(sorted=True)]
              for src, dst in indices]
    candidates = []
    for _ in range(k):
        for image_groups in groups:
            chosen = [queries[torch.randint(len(queries), (), device=device)]
                      for queries in image_groups]
            if chosen:
                chosen = torch.stack(chosen)
                candidates.append(chosen[torch.randperm(len(chosen), device=device)].tolist())
            else:
                candidates.append([])
    return candidates


def multiple_proposal_loss(criterion, outputs, targets):
    """Use the existing point/block loss path for repeated whole-object targets."""
    from detectron2.utils.events import get_event_storage
    dynamic = getattr(criterion, 'dynamic_assignment', False)
    min_iou = getattr(criterion, 'extra_proposal_min_iou', -1.0)
    correct_class = getattr(criterion, 'extra_proposal_require_correct_class', False)
    thresholds = getattr(criterion, 'extra_proposal_iou_thresholds', ())
    repeated = targets if dynamic else repeat_object_targets(targets, criterion.proposals_per_gt)
    indices = criterion.matcher(outputs, repeated)
    before_filter = sum(len(src) for src, _ in indices)
    if min_iou >= 0:
        indices = filter_extra_proposals(outputs, targets, indices, criterion.proposals_per_gt, min_iou, correct_class, thresholds)
    count = outputs['pred_masks'].new_tensor([sum(len(src) for src, _ in indices)])
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(count)
    count = (count / get_world_size()).clamp_min(1).item()
    losses = {}
    for depth, layer in enumerate([outputs] + outputs.get('aux_outputs', [])):
        layer_indices = indices if depth == 0 else criterion.matcher(layer, repeated)
        if min_iou >= 0 and depth > 0:
            layer_indices = filter_extra_proposals(layer, targets, layer_indices,
                                                    criterion.proposals_per_gt, min_iou, correct_class, thresholds)
        layer_count = count
        if (dynamic or min_iou >= 0) and depth > 0:
            matched = outputs['pred_masks'].new_tensor([sum(len(src) for src, _ in layer_indices)])
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(matched)
            layer_count = (matched / get_world_size()).clamp_min(1).item()
        for loss in criterion.losses:
            values = criterion.get_loss(loss, layer, repeated, layer_indices, layer_count)
            losses.update({key + ('' if depth == 0 else f'_{depth - 1}'): value
                           for key, value in values.items()})
    object_indices = indices if dynamic else [(src, dst // criterion.proposals_per_gt) for src, dst in indices]
    losses.update(criterion.loss_object_decoder(outputs, object_indices, targets))
    storage = get_event_storage()
    represented = sum(dst.unique().numel() for _, dst in object_indices)
    objects = sum(len(t['labels']) for t in targets)
    storage.put_scalar('multiple_proposals/matched_queries', sum(len(src) for src, _ in indices) / len(targets))
    storage.put_scalar('multiple_proposals/objects_without_match', (objects - represented) / len(targets))
    storage.put_scalar('multiple_proposals/queries_per_represented_object',
                       sum(len(src) for src, _ in indices) / max(1, represented))
    if min_iou >= 0:
        storage.put_scalar('multiple_proposals/filtered_queries',
                           (before_filter - sum(len(src) for src, _ in indices)) / len(targets))
    return losses
