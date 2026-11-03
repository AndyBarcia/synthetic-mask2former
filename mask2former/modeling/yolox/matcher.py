"""Dynamic-k assignment for dense things and mask matching for object targets."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou, masks_to_boxes
from .mask_ops import DynamicMaskRenderer
from scipy.optimize import linear_sum_assignment


class YOLOXMatcher(nn.Module, DynamicMaskRenderer):
    def __init__(self, num_classes, thing_ids, *, box_free=False, soft_labels=False,
                 mask_cost=True, object_iou_threshold=0.5):
        super().__init__()
        self.box_free = box_free
        self.soft_labels = soft_labels
        self.dynamic_masks = True
        self.mask_cost = mask_cost
        self.object_iou_threshold = object_iou_threshold
        lookup = torch.full((num_classes,), -1, dtype=torch.long)
        lookup[torch.tensor(sorted(thing_ids), dtype=torch.long)] = torch.arange(len(thing_ids))
        self.register_buffer("thing_lookup", lookup)

    @torch.no_grad()
    def forward(self, outputs, targets):
        raw, dense = outputs["dense_predictions"], outputs["mask_features"]
        indices = []
        for index, target in enumerate(targets):
            things = self.thing_lookup[target["labels"]] >= 0
            labels = self.thing_lookup[target["labels"][things]]
            if self.box_free:
                indices.append(self._mask_assignment(raw, dense, index, labels, target["masks"][things]))
            else:
                boxes = masks_to_boxes(target["full_masks"][things].bool())
                if len(boxes):
                    boxes[:, 2:] += 1
                indices.append(self._box_assignment(raw, dense, index, boxes, labels, target["masks"][things]))
        return indices

    @torch.no_grad()
    def match_objects(self, proposals, targets):
        return [self._match_objects(p, t) for p, t in zip(proposals, targets)]

    @torch.no_grad()
    def _box_assignment(self, out, dense, index, boxes, labels, masks):
        """Center-constrained dynamic-k assignment with optional mask Dice cost."""
        device = dense.device
        if not len(boxes):
            empty = torch.empty(0, device=device, dtype=torch.long)
            return empty, empty
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        soft = self.soft_labels
        near = ((out['centers'][None] - centers[:, None]).abs() < 2.5*out['strides'][None]).all(-1)
        if soft:
            inside = ((out['centers'][None] > boxes[:, None, :2]) & (out['centers'][None] < boxes[:, None, 2:])).all(-1)
            # Tiny GT boxes may contain no stride-8 center: keep their nearest-center neighborhood.
            near = torch.where(inside.any(-1)[:, None], inside, near)
        candidates = torch.where(near.any(0))[0]
        if not len(candidates):
            raise ValueError('No assignment candidates for nonempty valid boxes')
        iou = box_iou(boxes, out['boxes'][index, candidates]).clamp(0, 1)
        prob = (out['cls'][index, candidates].sigmoid() * out['obj'][index, candidates, None].sigmoid()).sqrt().clamp(1e-5, 1-1e-5)
        # BCE across classes without allocating [GT, anchors, classes].
        cls_cost = -torch.log1p(-prob).sum(-1)[None] - prob[:, labels].T.log() + torch.log1p(-prob[:, labels].T)
        cost = cls_cost - 3*(iou+1e-8).log() + (~near[:, candidates])*1e6
        if soft:
            scores = out['cls'][index, candidates]
            p = scores.sigmoid()
            negatives = F.softplus(scores)*p.square()
            positive_logits = scores[:, labels].T
            positive_prob = p[:, labels].T
            positive_cost = (F.softplus(positive_logits)-iou*positive_logits)*(iou-positive_prob).square()
            soft_cls = negatives.sum(-1)[None]-negatives[:, labels].T+positive_cost
            distance = ((out['centers'][candidates][None]-centers[:, None])/out['strides'][candidates][None]).square().sum(-1).sqrt()
            center_penalty = 10.**(distance-3).clamp(max=6)
            cost = soft_cls-3*(iou+1e-8).log()+center_penalty+(~near[:, candidates])*1e6
        if self.mask_cost:
            points = torch.randperm(dense.shape[-2]*dense.shape[-1], device=device)[:256]
            logits = self.render_candidates(out, dense, index, candidates, points).sigmoid()
            gt = masks.flatten(1)[:, points].float()
            dice = 1 - (2*gt @ logits.T+1)/(gt.sum(-1)[:, None]+logits.sum(-1)[None]+1)
            cost = cost + 3*dice
        match = torch.zeros_like(cost, dtype=torch.bool)
        dynamic_k = iou.topk(min(13 if soft else 10, len(candidates)), dim=1).values.sum(1).long().clamp_min(1)
        for gt in range(len(boxes)):
            match[gt, cost[gt].topk(int(dynamic_k[gt]), largest=False).indices] = True
        conflicts = match.sum(0) > 1
        if conflicts.any():
            best = cost[:, conflicts].argmin(0)
            match[:, conflicts] = False
            match[best, conflicts] = True
        selected = match.any(0)
        return candidates[selected], match[:, selected].float().argmax(0)

    @torch.no_grad()
    def _mask_assignment(self, out, dense, index, labels, masks):
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
        logits = self.render_candidates(out, dense, index, candidates, points)
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

    @torch.no_grad()
    def _match_objects(self, proposal, target):
        valid = target['masks'].flatten(1).any(1)
        gt = F.interpolate(target['masks'][valid, None].float(), proposal['masks'].shape[-2:], mode='nearest')[:, 0]
        if not len(gt):
            return torch.empty(0, device=gt.device, dtype=torch.long)
        predicted = (proposal['masks'] > 0).flatten(1).float()
        gt = gt.flatten(1)
        intersection = predicted @ gt.T
        iou = intersection / (predicted.sum(-1)[:, None] + gt.sum(-1)[None] - intersection).clamp_min(1)
        correct_class = proposal['labels'][:, None] == target['labels'][valid][None]
        rows, cols = linear_sum_assignment((-iou + 1000 * ~correct_class).cpu().numpy())
        rows = torch.as_tensor(rows, device=gt.device)
        cols = torch.as_tensor(cols, device=gt.device)
        return rows[correct_class[rows, cols] & (iou[rows, cols] > self.object_iou_threshold)]
