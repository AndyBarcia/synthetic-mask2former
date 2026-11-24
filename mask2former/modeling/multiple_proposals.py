"""Repeated whole-object matching slots and one-proposal-per-object teachers."""
import torch
from detectron2.utils.comm import get_world_size
from ..utils.misc import is_dist_avail_and_initialized


def repeat_object_targets(targets, copies):
    """Only whole objects enter matching; parts/fused annotations are excluded."""
    return [dict(labels=t['labels'].repeat_interleave(copies),
                 masks=t['masks'].repeat_interleave(copies, dim=0)) for t in targets]


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
    repeated = repeat_object_targets(targets, criterion.proposals_per_gt)
    indices = criterion.matcher(outputs, repeated)
    count = outputs['pred_masks'].new_tensor([sum(len(src) for src, _ in indices)])
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(count)
    count = (count / get_world_size()).clamp_min(1).item()
    losses = {}
    for depth, layer in enumerate([outputs] + outputs.get('aux_outputs', [])):
        layer_indices = indices if depth == 0 else criterion.matcher(layer, repeated)
        for loss in criterion.losses:
            values = criterion.get_loss(loss, layer, repeated, layer_indices, count)
            losses.update({key + ('' if depth == 0 else f'_{depth - 1}'): value
                           for key, value in values.items()})
    object_indices = [(src, dst // criterion.proposals_per_gt) for src, dst in indices]
    losses.update(criterion.loss_object_decoder(outputs, object_indices, targets))
    storage = get_event_storage()
    represented = sum(dst.unique().numel() for _, dst in object_indices)
    objects = sum(len(t['labels']) for t in targets)
    storage.put_scalar('multiple_proposals/matched_queries', sum(len(src) for src, _ in indices) / len(targets))
    storage.put_scalar('multiple_proposals/objects_without_match', (objects - represented) / len(targets))
    storage.put_scalar('multiple_proposals/queries_per_represented_object',
                       sum(len(src) for src, _ in indices) / max(1, represented))
    return losses
