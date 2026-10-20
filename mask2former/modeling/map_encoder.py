"""Independent last-mask-wins map encoder with call-local segment pooling."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class MapEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, dim_feedforward, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.inspection_cls = nn.Parameter(torch.randn(hidden_dim) * .02)
        self.inspection_void = nn.Parameter(torch.randn(hidden_dim) * .02)
        self.inspection_geometry = nn.Linear(3, hidden_dim)
        self.inspection_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward,
                dropout=0., batch_first=True), num_layers=2, enable_nested_tensor=False)

    @staticmethod
    def _pool_visible_segments(visible, features, cache):
        """Pool changed slots in one batched GEMM, then gather shared versions.

        visible: [image, state, segment, pixel]. Cache version zero carries the
        preceding call's final state. Comparisons are exact and fully on-device.
        """
        batch, states, slots, pixels = visible.shape
        width = features.shape[-1]
        prior_visible = cache.get("visible")
        prior_pooled = cache.get("pooled")
        if prior_visible is None:
            prior_visible = visible.new_zeros(batch, slots, pixels)
            prior_pooled = features.new_zeros(batch, slots, width)
        prior = torch.cat((prior_visible[:, None], visible[:, :-1]), dim=1)
        changed = (visible != prior).any(-1)
        # Flatten state/segment in temporal order, independently for each image.
        packed_index = changed.flatten(1).long().cumsum(1).reshape_as(changed) - 1
        count = changed.sum((1, 2))
        capacity = int(count.max())
        images, steps, segments = changed.nonzero(as_tuple=True)
        destinations = packed_index[images, steps, segments]
        weights = features.new_zeros(batch, capacity, pixels)
        weights[images, destinations] = visible[images, steps, segments].to(weights.dtype)
        values = torch.bmm(weights, features) / weights.sum(-1, keepdim=True).clamp_min(1)
        # Prefix maxima propagate the latest changed version in each slot.
        # Cached slot IDs precede every packed version, making zero-change
        # states (including empty masks) reference their cached value directly.
        slot_ids = torch.arange(slots, device=visible.device)[None, None]
        versions = torch.where(changed, packed_index + slots, slot_ids).cummax(1).values
        table = torch.cat((prior_pooled, values), dim=1)
        pooled = table.gather(1, versions.flatten(1)[..., None].expand(-1, -1, width))
        pooled = pooled.reshape(batch, states, slots, width)
        cache["visible"] = visible[:, -1].clone()
        cache["pooled"] = pooled[:, -1]
        return pooled

    def forward(self, mask_tokens, features, regions, previous, ancestors=None,
                positions=None, last_only=False, pool_cache=None):
        """Encode all independent prefix maps in a single batched encoder call."""
        # RL sampling and scoring may share an outer autocast context. Disabling
        # its parameter cache keeps grad-enabled scoring independent of sampling.
        with torch.autocast(device_type=mask_tokens.device.type,
                enabled=torch.is_autocast_enabled(mask_tokens.device.type), cache_enabled=False):
            return self._forward_impl(mask_tokens, features, regions, previous,
                ancestors, positions, last_only, {} if pool_cache is None else pool_cache)

    def _forward_impl(self, mask_tokens, features, regions, previous, ancestors,
                      positions, last_only, pool_cache):
        batch, steps = previous.shape
        device = previous.device
        indices = torch.arange(steps, device=device)
        nodes = indices[-1:] if last_only else indices
        context = (indices[None, None] <= nodes[None, :, None] if ancestors is None
                   else ancestors[:, nodes])
        tokens = previous[:, None].expand(-1, len(nodes), -1)
        valid = context & (tokens >= 0) & (tokens < self.num_queries)
        # Tree storage offsets differ from path depths. Geometry uses path rank.
        rank_source = (positions + 1 if positions is not None else
                       ancestors.sum(-1) if ancestors is not None else indices[None] + 1)
        ranks = torch.zeros(batch, len(nodes), self.num_queries, device=device, dtype=torch.int32)
        ranks.scatter_reduce_(2, tokens.clamp(0, self.num_queries - 1),
                              (rank_source[:, None] * valid).to(ranks.dtype), reduce='amax')
        painted_rank, owner = (regions[:, None].to(ranks.dtype) * ranks[..., None]).max(2)
        visible = owner[:, :, None] == torch.arange(self.num_queries, device=device)[None, None, :, None]
        visible &= painted_rank[:, :, None] > 0
        visible = torch.cat((visible, (painted_rank == 0)[:, :, None]), dim=2)
        area = visible.float().mean(-1)
        pooled = self._pool_visible_segments(visible, features, pool_cache)
        if "raw_area" not in pool_cache:
            pool_cache["raw_area"] = regions.float().mean(-1)
        raw_area = torch.cat((pool_cache["raw_area"][:, None].expand(-1, len(nodes), -1),
                              area[..., -1:]), dim=-1)
        order = F.pad(ranks.float() / ranks.amax(-1, keepdim=True).clamp_min(1), (0, 1))
        geometry = torch.stack((area, raw_area, order), dim=-1).to(mask_tokens.dtype)
        query_tokens = torch.cat((mask_tokens, self.inspection_void.expand(batch, 1, -1)), dim=1)
        region_tokens = query_tokens[:, None] + pooled + self.inspection_geometry(geometry)
        # Compact only the segment axis. Image and state axes stay batched.
        valid = area > 0
        capacity = int(valid.sum(-1).max())
        slots = valid.long().argsort(dim=-1, descending=True, stable=True)[..., :capacity]
        region_tokens = region_tokens.gather(2, slots[..., None].expand(-1, -1, -1, region_tokens.shape[-1]))
        region_tokens = region_tokens.flatten(0, 1)
        area = area.gather(2, slots).flatten(0, 1)
        tokens = torch.cat((self.inspection_cls.expand(len(region_tokens), 1, -1), region_tokens), dim=1)
        padding = F.pad(area == 0, (1, 0), value=False)

        def encode(tokens):
            with torch.autocast(device_type=tokens.device.type,
                    enabled=torch.is_autocast_enabled(tokens.device.type), cache_enabled=False):
                return self.inspection_encoder(tokens, src_key_padding_mask=padding)[:, 0]

        # Checkpoint attention/FFN activations once for the entire state batch.
        encoded = (checkpoint(encode, tokens, use_reentrant=False)
                   if self.training and torch.is_grad_enabled() else encode(tokens))
        encoded = encoded.reshape(batch, len(nodes), -1)
        if last_only:
            return F.pad(encoded, (0, 0, steps - 1, 0))
        return encoded
