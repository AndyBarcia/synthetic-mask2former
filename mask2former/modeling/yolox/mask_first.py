"""Box-free, NMS-free proposals: mask-center priors and mask-cost assignment."""
import torch
from torch.nn import functional as F
from .proposals import YOLOXMaskProposals, mask_losses


class MaskFirstProposals(YOLOXMaskProposals):
    def __init__(self, channels, num_classes, thing_ids, width=160, **kwargs):
        super().__init__(channels, num_classes, thing_ids, width, dynamic_masks=True, box_free=True, **kwargs)

    def proposals(self, out, dense, index=0, topk=100, nms=None, crop_margin=None):
        return super().proposals(out, dense, index, topk, nms, crop_margin)


@torch.no_grad()
def mask_assignment(head, out, dense, index, labels, masks):
    if not len(masks):
        empty = torch.empty(0, device=dense.device, dtype=torch.long)
        return empty, empty
    h, w = masks.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=dense.device), torch.arange(w, device=dense.device), indexing='ij')
    areas = masks.flatten(1).sum(-1).clamp_min(1)
    centers = torch.stack(((masks*(xx+.5)).sum((-2,-1))/areas*out['padded_size'][1]/w,
                           (masks*(yy+.5)).sum((-2,-1))/areas*out['padded_size'][0]/h), -1)
    distances = (out['centers'][None]-centers[:, None])/out['strides'][None]
    near = distances.abs().amax(-1) < 2.5
    candidates = torch.where(near.any(0))[0]
    # Include foreground points for small masks when constructing matching costs.
    points = [torch.randperm(h*w, device=dense.device)[:256]]
    for mask in masks:
        foreground = torch.where(mask.flatten() > 0)[0]
        if len(foreground): points.append(foreground[torch.randperm(len(foreground), device=dense.device)[:16]])
    points = torch.cat(points).unique()
    logits = head.render_candidates(out, dense, index, candidates, points)
    probability = logits.sigmoid(); target = masks.flatten(1)[:, points]
    intersection = target @ probability.T
    dice = 1-(2*intersection+1)/(target.sum(-1)[:, None]+probability.sum(-1)[None]+1)
    bce = (F.softplus(logits).sum(-1)[None]-target @ logits.T)/len(points)
    cls = (out['cls'][index, candidates].sigmoid()*out['obj'][index, candidates, None].sigmoid()).sqrt().clamp(1e-5, 1-1e-5)
    cls_cost = -torch.log1p(-cls).sum(-1)[None]-cls[:, labels].T.log()+torch.log1p(-cls[:, labels].T)
    cost = cls_cost + 5*dice + 2*bce + .1*distances[:, candidates].square().sum(-1).sqrt() + (~near[:, candidates])*1e6
    soft_iou = intersection/(target.sum(-1)[:, None]+probability.sum(-1)[None]-intersection).clamp_min(1)
    counts = soft_iou.topk(min(10, len(candidates)), dim=1).values.sum(-1).long().clamp(1, 10)
    match = torch.zeros_like(cost, dtype=torch.bool)
    for gt in range(len(masks)): match[gt, cost[gt].topk(int(counts[gt]), largest=False).indices] = True
    conflict = match.sum(0) > 1
    if conflict.any():
        best = cost[:, conflict].argmin(0); match[:, conflict] = False; match[best, conflict] = True
    valid = match.any(0)
    return candidates[valid], match[:, valid].float().argmax(0)


def mask_first_loss(head, out, dense, targets):
    total = dense.new_zeros(()); logs = {}
    lookup = torch.full((head.num_classes,), -1, device=dense.device, dtype=torch.long)
    lookup[head.thing_ids] = torch.arange(len(head.thing_ids), device=dense.device)
    for index, target in enumerate(targets):
        masks = F.interpolate(target['masks'][:, None].float(), dense.shape[-2:], mode='nearest')[:, 0]
        valid = masks.flatten(1).any(-1); masks, labels = masks[valid], target['labels'][valid]
        things = lookup[labels] >= 0
        ids, matched = mask_assignment(head, out, dense, index, lookup[labels[things]], masks[things])
        norm = max(len(ids), 1)
        objectness = torch.zeros_like(out['obj'][index]); objectness[ids] = 1
        losses = dict(obj=F.binary_cross_entropy_with_logits(out['obj'][index], objectness, reduction='sum')/norm)
        with torch.no_grad():
            binary = head.render_candidates(out, dense, index, ids) > 0
            gt = masks[things][matched].flatten(1)
            intersection = (binary*gt).sum(-1)
            quality = intersection/(binary.sum(-1)+gt.sum(-1)-intersection).clamp_min(1)
        classes = torch.zeros_like(out['cls'][index, ids])
        classes[torch.arange(len(ids), device=dense.device), lookup[labels[things]][matched]] = quality.sqrt().clamp_min(.1)
        losses['cls'] = F.binary_cross_entropy_with_logits(out['cls'][index, ids], classes, reduction='sum')/norm
        points = (torch.arange(dense.shape[-2]*dense.shape[-1], device=dense.device) if head.full_mask_loss else
                  torch.randperm(dense.shape[-2]*dense.shape[-1], device=dense.device)[:2048])
        predicted = head.render_candidates(out, dense, index, ids, points)
        bce, dice = mask_losses(predicted, masks[things][matched].flatten(1)[:, points])
        losses.update(mask_bce=5*bce, mask_dice=5*dice)
        stuff_targets = dense.new_zeros((len(head.stuff_ids), *dense.shape[-2:]))
        for k, category in enumerate(head.stuff_ids):
            rows = masks[labels == category]
            if len(rows): stuff_targets[k] = rows.amax(0)
        present = stuff_targets.flatten(1).any(-1)
        losses['stuff_presence'] = F.binary_cross_entropy_with_logits(out['stuff_presence'][index], present.float())
        stuff = head.render(out['stuff_kernels'][index, present], dense[index])[:, points]
        bce, dice = mask_losses(stuff, stuff_targets[present].flatten(1)[:, points])
        losses['stuff_mask'] = 5*(bce+dice)
        total = total+sum(losses.values())/len(targets)
        for key, value in losses.items(): logs[key] = logs.get(key, 0.)+float(value.detach())/len(targets)
        logs['positive_anchors'] = logs.get('positive_anchors', 0.)+len(ids)/len(targets)
        logs['matched_thing_fraction'] = logs.get('matched_thing_fraction', 0.)+len(matched.unique())/max(int(things.sum()),1)/len(targets)
    return total, logs
