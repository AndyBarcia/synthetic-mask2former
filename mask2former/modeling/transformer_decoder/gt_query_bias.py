"""Mask-conditioned priors over query identities."""
import torch
from torch import nn
from torch.nn import functional as F


class GTQueryBias(nn.Module):
    """Encode each mask's pooled image features and relative area into Q logits."""

    def __init__(self, feature_dim, hidden_dim, num_queries):
        super().__init__()
        self.gt_encoder = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.query_logits = nn.Linear(hidden_dim, num_queries)
        nn.init.zeros_(self.query_logits.weight)
        nn.init.zeros_(self.query_logits.bias)

    def forward(self, features, masks):
        """Return (number of masks, Q), including for empty GT sets.

        Area interpolation preserves small masks when pooling at feature resolution.
        Features are C,H,W; masks are N,H,W on the padded image canvas.
        """
        masks = masks.to(device=features.device, dtype=torch.float32)
        if masks.shape[0] == 0:
            pooled = features.new_zeros((0, features.shape[0]))
            area = features.new_zeros((0, 1))
        else:
            # Keep spatial reductions in FP32 under mixed precision.
            with torch.autocast(device_type=features.device.type, enabled=False):
                area = masks.flatten(1).mean(1, keepdim=True)
                resized = F.interpolate(
                    masks[:, None], size=features.shape[-2:], mode="area"
                )[:, 0].flatten(1)
                mass = resized.sum(1, keepdim=True)
                pooled = resized @ features.float().flatten(1).transpose(0, 1)
                pooled = pooled / mass.clamp_min(torch.finfo(torch.float32).eps)
        return self.query_logits(self.gt_encoder(torch.cat((pooled, area), dim=1).to(features.dtype)))

    def predicted_query_logits(self, features, mask_logits):
        # A predicted mask stands in for GT at inference. Score its own query.
        return torch.stack([
            self(image_features, masks.detach().sigmoid()).diagonal()
            for image_features, masks in zip(features, mask_logits)
        ])
