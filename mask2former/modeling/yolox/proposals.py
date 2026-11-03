"""Dense YOLOX-inspired mask proposals and their supervised training objective.

The shared pixel-decoder features provide a stride-4 mask basis and three
pyramid levels. Only score-selected masks are rendered; boxes supervise things
but never crop or suppress proposals passed to the object decoder.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from .mask_ops import DynamicMaskRenderer


def block(cin, cout, kernel=3):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU())


def xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), -1)


class YOLOXMaskProposals(nn.Module, DynamicMaskRenderer):
    def __init__(self, channels, num_classes, thing_ids, width=96, *, soft_labels=False, large_kernel=False, box_free=False):
        super().__init__()
        self.num_classes = num_classes
        self.soft_labels = soft_labels
        self.large_kernel = large_kernel
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
        results['dynamic_params'] = []
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
            results['dynamic_params'].append(flat(self.dynamic[level](reg)))
        out = {k: torch.cat(v, 0 if k in ('centers', 'strides') else 1) for k, v in results.items()}
        context = dense.mean((-2, -1))
        out['stuff_kernels'] = self.stuff_kernel(self.stuff_tokens[None] + self.stuff_context(context)[:, None])
        out['stuff_presence'] = self.stuff_presence(context)
        out['padded_size'] = tuple(padded_size)
        out['dynamic_features'] = self.mask_features(dense)
        return out

    def proposals(self, out, dense, index=0, topk=100):
        cls_score, local_labels = out['cls'][index].sigmoid().max(-1)
        score = cls_score * out['obj'][index].sigmoid()
        pre = score.topk(min(topk, len(score))).indices
        kernels = torch.cat((out['kernels'][index, pre], out['stuff_kernels'][index]))
        scores = torch.cat((score[pre], out['stuff_presence'][index].sigmoid()))
        labels = torch.cat((self.thing_ids[local_labels[pre]], self.stuff_ids))
        canvas = dense.new_tensor((0, 0, out['padded_size'][1], out['padded_size'][0]))
        boxes = None if self.box_free else torch.cat((out['boxes'][index, pre], canvas[None].expand(len(self.stuff_ids), -1)))
        isthing = torch.cat((torch.ones(len(pre), device=dense.device, dtype=torch.bool),
                             torch.zeros(len(self.stuff_ids), device=dense.device, dtype=torch.bool)))
        keep = scores.topk(min(topk, len(scores))).indices
        masks = self.render(kernels[keep], dense[index]).reshape(-1, *dense.shape[-2:])
        positions = torch.where(keep < len(pre))[0]
        masks[positions] = self.render_candidates(out, dense, index, pre[keep[positions]]).reshape(-1, *dense.shape[-2:])
        dense_ids = torch.cat((pre, pre.new_full((len(self.stuff_ids),), -1)))[keep]
        result = dict(dense_ids=dense_ids, masks=masks, labels=labels[keep], scores=scores[keep], embeddings=kernels[keep, :-1], isthing=isthing[keep],
                      bias=kernels[keep, -1:],
                      centers=torch.cat((out['centers'][pre], canvas[None, 2:].expand(len(self.stuff_ids), -1)/2))[keep],
                      strides=torch.cat((out['strides'][pre], canvas[None, 2:].expand(len(self.stuff_ids), -1)))[keep])
        if boxes is not None: result['boxes'] = boxes[keep]
        result['dynamic_parameters'] = torch.cat((out['dynamic_params'][index, pre],
                                                   kernels.new_zeros((len(self.stuff_ids), 169))))[keep]
        return result
