"""Hard painted-segment assignment with differentiable selected-query losses."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def selected_segments(mask_logits, class_logits, order, thing_class_ids, size, reverse=False):
    """Return visible query groups and masks, merging stuff after last-wins painting."""
    # A query's final occurrence wins; retain its final position in paint order.
    order = list(reversed(order)) if reverse else list(order)
    order = [q for i, q in enumerate(order) if q not in order[i + 1:]]
    if not order:
        return order, [], mask_logits.new_zeros((0, *size))
    masks = F.interpolate(mask_logits[order, None].float(), size=size,
                          mode='bilinear', align_corners=False)[:, 0] > 0
    ranks = torch.arange(1, len(order) + 1, device=masks.device)
    rank, winner = (masks * ranks[:, None, None]).max(0)
    labels = class_logits[..., :-1].argmax(-1)
    things = set(thing_class_ids)
    groups = {}
    for position, query in enumerate(order):
        visible = (rank > 0) & (winner == position)
        if not visible.any():
            continue
        category = int(labels[query])
        key = ('thing', query) if category in things else ('stuff', category)
        if key not in groups:
            groups[key] = ([], torch.zeros_like(visible))
        groups[key][0].append(query)
        groups[key][1].logical_or_(visible)
    return order, [g[0] for g in groups.values()], torch.stack(
        [g[1] for g in groups.values()]
    ).float() if groups else mask_logits.new_zeros((0, *size))


def segment_predictions(outputs, image, order, groups):
    """Soft last-wins compositing; groups follow the final hard panoptic map."""
    masks = outputs['pred_masks'][image]
    if groups:
        probabilities = masks[order].float().sigmoid()
        # A query contributes where no subsequently painted query covers it.
        visible = []
        transmittance = torch.ones_like(probabilities[0])
        for probability in probabilities.flip(0):
            visible.append(probability * transmittance)
            transmittance = transmittance * (1 - probability)
        visible = torch.stack(list(reversed(visible)))
        positions = {query: i for i, query in enumerate(order)}
        segment_probs = torch.stack([
            visible[[positions[q] for q in group]].sum(0) for group in groups
        ])
        segment_masks = torch.logit(segment_probs.clamp(1e-6, 1 - 1e-6))
        classes = torch.stack([outputs['pred_logits'][image, group].mean(0) for group in groups])
    else:
        segment_masks = masks[:0]
        classes = outputs['pred_logits'][image, :0]
    result = {'pred_masks': segment_masks[None], 'pred_logits': classes[None]}
    if 'gt_query_bias_logits' in outputs:
        logits = outputs['gt_query_bias_logits'][image]
        result['gt_query_bias_logits'] = [torch.stack([
            logits[:, group].mean(-1) for group in groups
        ], -1) if groups else logits[:, :0]]
    elif 'query_bias_logits' in outputs:
        logits = outputs['query_bias_logits'][image]
        result['query_bias_logits'] = (torch.stack([
            logits[group].mean() for group in groups
        ]) if groups else logits[:0])[None]
    return result
