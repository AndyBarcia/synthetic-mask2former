"""Dense YOLOX-inspired mask proposals and their supervised training objective.

The shared pixel-decoder features provide a stride-4 mask basis and three
pyramid levels. Only score-selected masks are rendered; boxes supervise things
but never crop or suppress proposals passed to the object decoder.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou, masks_to_boxes


def block(cin, cout, kernel=3):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU())


def xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), -1)


class YOLOXMaskProposals(nn.Module):
    def __init__(self, channels, num_classes, thing_ids, width=96, *, soft_labels=False, dynamic_masks=False, large_kernel=False, full_mask_loss=False, box_free=False):
        super().__init__()
        self.num_classes = num_classes
        self.soft_labels = soft_labels
        self.dynamic_masks = dynamic_masks
        self.large_kernel = large_kernel
        self.full_mask_loss = full_mask_loss
        self.box_free = box_free
        self.register_buffer('thing_ids', torch.tensor(sorted(thing_ids), dtype=torch.long))
        self.register_buffer('stuff_ids', torch.tensor(sorted(set(range(num_classes)) - set(thing_ids)), dtype=torch.long))
        nt, ns = len(self.thing_ids), len(self.stuff_ids)
        self.stems = nn.ModuleList([block(channels + 2, width, 1) for _ in range(3)])
        self.cls_towers = nn.ModuleList([nn.Sequential(block(width, width), block(width, width)) for _ in range(3)])
        self.reg_towers = nn.ModuleList([nn.Sequential(block(width, width), block(width, width)) for _ in range(3)])
        self.cls = nn.ModuleList([nn.Conv2d(width, nt, 1) for _ in range(3)])
        self.obj = nn.ModuleList([nn.Conv2d(width, 1, 1) for _ in range(3)])
        self.box = nn.ModuleList([] if box_free else [nn.Conv2d(width, 4, 1) for _ in range(3)])
        self.kernel = nn.ModuleList([nn.Conv2d(width, channels + 1, 1) for _ in range(3)])
        if large_kernel:
            # Shared context operator across pyramid levels; residual keeps local features.
            self.context = nn.Sequential(nn.Conv2d(width, width, 5, padding=2, groups=width, bias=False),
                                         nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, width, 1))
        if dynamic_masks:
            self.mask_features = nn.Sequential(block(channels, 32), nn.Conv2d(32, 8, 1))
            # Three dynamic 1x1 layers: (8 features + 2 relative coordinates)->8->8->1.
            self.dynamic = nn.ModuleList([nn.Conv2d(width, 169, 1) for _ in range(3)])
            for layer in self.dynamic:
                nn.init.normal_(layer.weight, std=.01); nn.init.zeros_(layer.bias)
        self.stuff_tokens = nn.Parameter(torch.randn(ns, channels) * .02)
        self.stuff_context = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.stuff_kernel = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels + 1))
        self.stuff_presence = nn.Linear(channels, ns)
        for layer in [*self.cls, *self.obj, self.stuff_presence]:
            nn.init.constant_(layer.bias, math.log(.01 / .99))
        for layer in self.box:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        for layer in [*self.kernel, self.stuff_kernel[-1]]:
            nn.init.normal_(layer.weight, std=.001); nn.init.zeros_(layer.bias)

    def forward(self, multi, dense, padded_size):
        results = {k: [] for k in ['cls', 'obj', 'boxes', 'kernels', 'centers', 'strides']}
        if self.box_free: del results['boxes']
        if self.dynamic_masks: results['dynamic_params'] = []
        # Pixel decoder returns coarse-to-fine; order by spatial size explicitly.
        for level, feature in enumerate(sorted(multi, key=lambda x: x.shape[-2] * x.shape[-1], reverse=True)):
            b, _, h, w = feature.shape
            sy, sx = padded_size[0] / h, padded_size[1] / w
            yy, xx = torch.meshgrid(torch.arange(h, device=dense.device), torch.arange(w, device=dense.device), indexing='ij')
            grid = torch.stack((xx, yy), -1).reshape(-1, 2).float()
            stride = dense.new_tensor((sx, sy))[None].expand(h*w, -1)
            coords = torch.stack(((xx+.5)/w*2-1, (yy+.5)/h*2-1))[None].expand(b, -1, -1, -1)
            stem = self.stems[level](torch.cat((feature, coords), 1))
            if self.large_kernel: stem = stem + self.context(stem)
            cls, reg = self.cls_towers[level](stem), self.reg_towers[level](stem)
            flat = lambda x: x.flatten(2).transpose(1, 2)
            if not self.box_free:
                raw = flat(self.box[level](reg))
                box = torch.cat(((raw[..., :2] + grid) * stride,
                                 raw[..., 2:].clamp(-6, 6).exp() * stride), -1)
                results['boxes'].append(xyxy(box))
            for key, value in [('cls', flat(self.cls[level](cls))), ('obj', flat(self.obj[level](reg))[..., 0]),
                               ('kernels', flat(self.kernel[level](reg)))]:
                results[key].append(value)
            results['centers'].append((grid+.5)*stride); results['strides'].append(stride)
            if self.dynamic_masks: results['dynamic_params'].append(flat(self.dynamic[level](reg)))
        out = {k: torch.cat(v, 0 if k in ('centers', 'strides') else 1) for k, v in results.items()}
        context = dense.mean((-2, -1))
        out['stuff_kernels'] = self.stuff_kernel(self.stuff_tokens[None] + self.stuff_context(context)[:, None])
        out['stuff_presence'] = self.stuff_presence(context)
        out['padded_size'] = tuple(padded_size)
        if self.dynamic_masks: out['dynamic_features'] = self.mask_features(dense)
        return out

    @staticmethod
    def render(kernels, features):
        return kernels[..., :-1] @ features.flatten(1) + kernels[..., -1:]

    def render_candidates(self, out, dense, index, ids, points=None):
        features = dense[index].flatten(1)
        if points is not None: features = features[:, points]
        kernels = out['kernels'][index, ids]
        base = kernels[..., :-1] @ features + kernels[..., -1:]
        if not self.dynamic_masks: return base
        h, w = dense.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h, device=dense.device), torch.arange(w, device=dense.device), indexing='ij')
        locations = torch.stack(((xx+.5)*out['padded_size'][1]/w, (yy+.5)*out['padded_size'][0]/h)).flatten(1)
        prototypes = out['dynamic_features'][index].flatten(1)
        if points is not None: locations = locations[:, points]; prototypes = prototypes[:, points]
        # Normalize relative positions by feature-level scale, as in compact dynamic heads.
        relative = (locations[None]-out['centers'][ids, :, None])/(out['strides'][ids, :, None]*8)
        x = torch.cat((prototypes[None].expand(len(ids), -1, -1), relative), 1)
        parameters = out['dynamic_params'][index, ids]
        offset = 0
        for cin, cout in [(10, 8), (8, 8), (8, 1)]:
            count = cin*cout
            weight = parameters[:, offset:offset+count].reshape(-1, cout, cin); offset += count
            bias = parameters[:, offset:offset+cout, None]; offset += cout
            x = torch.bmm(weight, x)+bias
            if cout != 1: x = F.relu(x)
        return base+x[:, 0]

    def proposals(self, out, dense, index=0, topk=100, nms=None, crop_margin=None):
        if nms is not None or crop_margin is not None:
            raise ValueError('Object-decoder proposals do not use NMS or box cropping')
        cls_score, local_labels = out['cls'][index].sigmoid().max(-1)
        score = cls_score * out['obj'][index].sigmoid()
        # Cheap prefilter bounds NMS work; there is no hard confidence cutoff.
        if self.box_free:
            if nms is not None or crop_margin is not None: raise ValueError('mask-first proposals forbid NMS and box cropping')
        pre = score.topk(min(topk if nms is None else 1500, len(score))).indices
        kernels = torch.cat((out['kernels'][index, pre], out['stuff_kernels'][index]))
        scores = torch.cat((score[pre], out['stuff_presence'][index].sigmoid()))
        labels = torch.cat((self.thing_ids[local_labels[pre]], self.stuff_ids))
        canvas = dense.new_tensor((0, 0, out['padded_size'][1], out['padded_size'][0]))
        boxes = None if self.box_free else torch.cat((out['boxes'][index, pre], canvas[None].expand(len(self.stuff_ids), -1)))
        isthing = torch.cat((torch.ones(len(pre), device=dense.device, dtype=torch.bool),
                             torch.zeros(len(self.stuff_ids), device=dense.device, dtype=torch.bool)))
        keep = scores.topk(min(topk, len(scores))).indices
        masks = self.render(kernels[keep], dense[index]).reshape(-1, *dense.shape[-2:])
        if self.dynamic_masks:
            positions = torch.where(keep < len(pre))[0]
            masks[positions] = self.render_candidates(out, dense, index, pre[keep[positions]]).reshape(-1, *dense.shape[-2:])
        result = dict(masks=masks, labels=labels[keep], scores=scores[keep], embeddings=kernels[keep, :-1], isthing=isthing[keep],
                      bias=kernels[keep, -1:],
                      centers=torch.cat((out['centers'][pre], canvas[None, 2:].expand(len(self.stuff_ids), -1)/2))[keep],
                      strides=torch.cat((out['strides'][pre], canvas[None, 2:].expand(len(self.stuff_ids), -1)))[keep])
        if boxes is not None: result['boxes'] = boxes[keep]
        if self.dynamic_masks:
            result['dynamic_parameters'] = torch.cat((out['dynamic_params'][index, pre],
                                                       kernels.new_zeros((len(self.stuff_ids), 169))))[keep]
        return result


@torch.no_grad()
def dynamic_assignment(out, index, boxes, labels, masks, features, mask_cost=True, head=None, dense=None):
    """Center-constrained dynamic-k assignment with optional mask Dice cost."""
    device = features.device
    if not len(boxes):
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    soft = head is not None and head.soft_labels
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
    if mask_cost:
        points = torch.randperm(features.shape[-2]*features.shape[-1], device=device)[:256]
        logits = (head.render_candidates(out, dense, index, candidates, points) if head is not None else
                  YOLOXMaskProposals.render(out['kernels'][index, candidates], features)[:, points]).sigmoid()
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


def mask_losses(logits, targets):
    if not len(logits):
        return logits.sum()*0, logits.sum()*0
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none').mean(-1).mean()
    p = logits.sigmoid(); target = targets.float()
    dice = (1-(2*(p*target).sum(-1)+1)/(p.sum(-1)+target.sum(-1)+1)).mean()
    return bce, dice


def proposal_loss(head, out, dense, targets, mask_cost=True):
    """Supervise things with detection+mask losses, stuff with presence+mask losses."""
    total = dense.new_zeros(()); logs = {}; nt = len(head.thing_ids)
    global_to_local = torch.full((head.num_classes,), -1, device=dense.device, dtype=torch.long)
    global_to_local[head.thing_ids] = torch.arange(nt, device=dense.device)
    for index, target in enumerate(targets):
        # Padding/cropping may leave empty annotated segments: remove them.
        valid = target['masks'].flatten(1).any(1)
        labels, full_masks = target['labels'][valid], target['masks'][valid]
        masks = F.interpolate(full_masks[:, None].float(), dense.shape[-2:], mode='nearest')[:, 0]
        isthing = global_to_local[labels] >= 0
        boxes = masks_to_boxes(full_masks[isthing].bool())
        if len(boxes): boxes[:, 2:] += 1  # exclusive maximum, including one-pixel objects
        positives, matched = dynamic_assignment(out, index, boxes, global_to_local[labels[isthing]], masks[isthing], dense[index], mask_cost, head, dense)
        norm = max(len(positives), 1)
        obj_target = torch.zeros_like(out['obj'][index]); obj_target[positives] = 1
        losses = dict(obj=F.binary_cross_entropy_with_logits(out['obj'][index], obj_target, reduction='sum')/norm)
        cls_target = torch.zeros_like(out['cls'][index, positives])
        if len(positives):
            pbox, gbox = out['boxes'][index, positives], boxes[matched]
            lt, rb = torch.maximum(pbox[:, :2], gbox[:, :2]), torch.minimum(pbox[:, 2:], gbox[:, 2:])
            inter = (rb-lt).clamp_min(0).prod(-1)
            iou = inter/((pbox[:, 2:]-pbox[:, :2]).clamp_min(0).prod(-1)+(gbox[:, 2:]-gbox[:, :2]).prod(-1)-inter).clamp_min(1e-8)
            losses['box'] = 5*(1-iou.square()).mean()
            cls_target[torch.arange(len(positives), device=dense.device), global_to_local[labels[isthing]][matched]] = iou.detach()
        else:
            losses['box'] = out['boxes'][index].sum()*0
        losses['cls'] = F.binary_cross_entropy_with_logits(out['cls'][index, positives], cls_target, reduction='sum')/norm
        if head.soft_labels:
            all_targets = torch.zeros_like(out['cls'][index]); all_targets[positives] = cls_target
            cls_logits = out['cls'][index]
            losses['cls'] = (F.binary_cross_entropy_with_logits(cls_logits, all_targets, reduction='none') *
                             (all_targets-cls_logits.sigmoid()).abs().square()).sum()/norm
        points = (torch.arange(dense.shape[-2]*dense.shape[-1], device=dense.device) if head.full_mask_loss else
                  torch.randperm(dense.shape[-2]*dense.shape[-1], device=dense.device)[:2048])
        kernels = out['kernels'][index, positives]
        pmasks = head.render_candidates(out, dense, index, positives, points)
        gt_masks = masks[isthing][matched].flatten(1)[:, points]
        bce, dice = mask_losses(pmasks, gt_masks)
        losses['mask_bce'], losses['mask_dice'] = 5*bce, 5*dice
        stuff_targets = torch.zeros((len(head.stuff_ids), *dense.shape[-2:]), device=dense.device)
        for k, category in enumerate(head.stuff_ids):
            rows = masks[labels == category]
            if len(rows): stuff_targets[k] = rows.amax(0)
        present = stuff_targets.flatten(1).any(1)
        losses['stuff_presence'] = F.binary_cross_entropy_with_logits(out['stuff_presence'][index], present.float())
        skernels = out['stuff_kernels'][index, present]
        slogits = head.render(skernels, dense[index])[:, points]
        sbce, sdice = mask_losses(slogits, stuff_targets[present].flatten(1)[:, points])
        losses['stuff_mask'] = 5*(sbce+sdice)
        total = total + sum(losses.values()) / len(targets)
        for key, value in losses.items(): logs[key] = logs.get(key, 0.) + float(value.detach())/len(targets)
        logs['positive_anchors'] = logs.get('positive_anchors', 0.) + len(positives)/len(targets)
        logs['matched_thing_fraction'] = logs.get('matched_thing_fraction', 0.) + len(matched.unique())/max(int(isthing.sum()), 1)/len(targets)
    return total, logs
