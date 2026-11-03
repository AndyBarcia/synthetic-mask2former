"""Shared dynamic-mask rendering operations used by the head and matcher."""
import torch
from torch.nn import functional as F


class DynamicMaskRenderer:
    @staticmethod
    def upsample_mask_logits(logits, size):
        if not len(logits):
            return logits.new_empty((0, *size)) + logits.sum() * 0
        return F.interpolate(logits[:, None], size=size, mode='bilinear', align_corners=False)[:, 0]

    def render_candidates_at_resolution(self, out, dense, index, ids, size):
        logits = self.render_candidates(out, dense, index, ids).reshape(-1, *dense.shape[-2:])
        return self.upsample_mask_logits(logits, size).flatten(1)

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
