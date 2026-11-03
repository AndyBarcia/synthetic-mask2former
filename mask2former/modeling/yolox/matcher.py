"""Dynamic-k assignment for dense things and mask matching for object targets."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou, masks_to_boxes
from .mask_ops import DynamicMaskRenderer
from ..matcher import HungarianMatcher


class YOLOXMatcher(nn.Module, DynamicMaskRenderer):
    def __init__(self, num_classes, thing_ids, *, box_free=False, soft_labels=False,
                 mask_cost=True, object_iou_threshold=0.5,
                 cost_class=2., cost_mask=5., cost_dice=5., num_points=1024,
                 ensure_gt_coverage=False):
        super().__init__()
        self.box_free = box_free
        self.soft_labels = soft_labels
        self.dynamic_masks = True
        self.mask_cost = mask_cost
        self.ensure_gt_coverage = ensure_gt_coverage
        # object_iou_threshold is accepted for old configs/checkpoints only.
        # Teacher targets retain every Hungarian assignment, as in MaskFormer.
        self.object_matcher = HungarianMatcher(cost_class=cost_class, cost_mask=cost_mask,
                                              cost_dice=cost_dice, num_points=num_points,
                                              mask_loss_type='point')
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
            points = torch.randperm(masks.shape[-2]*masks.shape[-1], device=device)[:256]
            logits = self.render_candidates_at_resolution(out, dense, index, candidates, masks.shape[-2:])[:, points].sigmoid()
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
        return self._finish_assignment(match, cost, candidates)

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
        logits = self.render_candidates_at_resolution(out, dense, index, candidates, masks.shape[-2:])[:, points]
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
        return self._finish_assignment(match, cost, candidates)

    def _finish_assignment(self, match, cost, candidates):
        if self.ensure_gt_coverage:
            # Repair dynamic-k conflicts without a second/Hungarian assignment.
            # Prefer unused candidates; otherwise transfer one from a GT with >1.
            if match.shape[1] < match.shape[0]:
                raise ValueError('Not enough candidates to cover every GT')
            for gt in torch.where(~match.any(1))[0].tolist():
                counts = match.sum(1)
                available = ~match.any(0) | (match & (counts > 1)[:, None]).any(0)
                choices = torch.where(available)[0]
                query = choices[cost[gt, choices].argmin()]
                match[:, query] = False
                match[gt, query] = True
        selected = match.any(0)
        ids = candidates[selected]
        matched = match[:, selected].float().argmax(0)
        if self.ensure_gt_coverage:
            # First member of each GT's group is its minimum-cost teacher token.
            ranking = cost[matched, torch.where(selected)[0]].argsort()
            ids, matched = ids[ranking], matched[ranking]
        return ids, matched

    @torch.no_grad()
    def _match_objects(self, proposal, target):
        valid = target['masks'].flatten(1).any(1)
        targets = [dict(labels=target['labels'][valid], masks=target['masks'][valid])]
        # HungarianMatcher ignores its final legacy background channel.
        logits = proposal['class_logits']
        logits = torch.cat((logits, logits.new_zeros((len(logits), 1))), dim=-1)
        indices = self.object_matcher(dict(pred_logits=logits[None],
                                          pred_masks=proposal['masks'][None]), targets)
        return indices[0][0].to(device=proposal['masks'].device)
