"""Independent last-mask-wins map encoder with call-local segment pooling."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .state_cache import StateCache, pack_by_image


class MapEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, dim_feedforward, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.inspection_cls = nn.Parameter(torch.randn(hidden_dim) * .02)
        self.inspection_void = nn.Parameter(torch.randn(hidden_dim) * .02)
        self.inspection_geometry = nn.Linear(2, hidden_dim)
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

    def render_maps(self, regions, previous, ancestors=None, positions=None, last_only=False):
        """Canonical query ownership. Paint order matters only where masks overlap."""
        batch, steps = previous.shape
        device = previous.device
        if ancestors is None and positions is None:
            # A temporal last-paint scan avoids the [B, states, queries, pixels]
            # rank product used for independently branched map states.
            pixels = regions.shape[-1]
            valid = (previous >= 0) & (previous < self.num_queries)
            masks = regions.gather(1, previous.clamp(0, self.num_queries - 1)[..., None]
                                   .expand(-1, -1, pixels)) & valid[..., None]
            ranks = torch.arange(1, steps + 1, device=device, dtype=torch.int32)
            painted = masks.to(torch.int32) * ranks[None, :, None]
            latest = (painted.amax(1, keepdim=True) if last_only else painted.cummax(1).values)
            owner = previous.gather(1, (latest.long() - 1).clamp_min(0).flatten(1)).reshape_as(latest)
            return owner.masked_fill(latest == 0, self.num_queries)
        indices = torch.arange(steps, device=device)
        nodes = indices[-1:] if last_only else indices
        context = (indices[None, None] <= nodes[None, :, None] if ancestors is None
                   else ancestors[:, nodes])
        tokens = previous[:, None].expand(-1, len(nodes), -1)
        valid = context & (tokens >= 0) & (tokens < self.num_queries)
        rank_source = (positions + 1 if positions is not None else
                       ancestors.sum(-1) if ancestors is not None else indices[None] + 1)
        ranks = torch.zeros(batch, len(nodes), self.num_queries, device=device, dtype=torch.int32)
        ranks.scatter_reduce_(2, tokens.clamp(0, self.num_queries - 1),
                              (rank_source[:, None] * valid).to(ranks.dtype), reduce='amax')
        painted_rank, owner = (regions[:, None].to(ranks.dtype) * ranks[..., None]).max(2)
        return owner.masked_fill(painted_rank == 0, self.num_queries)

    def forward(self, mask_tokens, features, regions, previous, ancestors=None,
                positions=None, last_only=False, pool_cache=None, *, state_cache=None,
                image_ids=None, merge_states=True):
        """Encode each distinct ownership map once; gather all original prefixes."""
        with torch.autocast(device_type=mask_tokens.device.type,
                enabled=torch.is_autocast_enabled(mask_tokens.device.type), cache_enabled=False):
            owner = self.render_maps(regions, previous, ancestors, positions, last_only)
            encoded = self.encode_maps(mask_tokens, features, regions, owner, pool_cache,
                state_cache=state_cache, image_ids=image_ids, merge_states=merge_states)
            return F.pad(encoded, (0, 0, previous.shape[1] - 1, 0)) if last_only else encoded

    def encode_maps(self, mask_tokens, features, regions, owner, pool_cache=None, *,
                    state_cache=None, image_ids=None, merge_states=True, unique_states=False):
        batch, states, _ = owner.shape
        device = owner.device
        if image_ids is None:
            image_ids = torch.arange(batch, device=device)
        else:
            image_ids = image_ids.to(device=device, dtype=torch.long)
            if image_ids.shape != (batch,):
                raise ValueError("image_ids must contain one stable source ID per batch row")
        memo = (StateCache() if state_cache is None else state_cache) if merge_states else None
        if memo is not None:
            keys = torch.cat((image_ids[:, None, None].expand(-1, states, 1), owner), -1).flatten(0, 1)
            if unique_states:
                missing = torch.arange(batch * states, device=device)
            else:
                lookup = memo.lookup(keys)
                missing = lookup.missing
            if not len(missing):
                return memo.store(lookup, None).reshape(batch, states, -1)
        else:
            missing = torch.arange(batch * states, device=device)
        images, nodes = missing // states, missing % states
        if len(missing) == batch * states:
            active_owner, active_slots = owner, nodes
        else:
            active_slots, capacity = pack_by_image(images, batch)
            active_owner = owner.new_full((batch, capacity, owner.shape[-1]), self.num_queries)
            active_owner[images, active_slots] = owner[images, nodes]
        pool_cache = {} if pool_cache is None else pool_cache
        visible = active_owner[:, :, None] == torch.arange(self.num_queries + 1, device=device)[None, None, :, None]
        area = visible.float().mean(-1)
        pooled = self._pool_visible_segments(visible, features, pool_cache)
        if "raw_area" not in pool_cache:
            pool_cache["raw_area"] = regions.float().mean(-1)
        raw_area = torch.cat((pool_cache["raw_area"][:, None].expand(-1, active_owner.shape[1], -1),
                              area[..., -1:]), dim=-1)
        geometry = torch.stack((area, raw_area), dim=-1).to(mask_tokens.dtype)
        query_tokens = torch.cat((mask_tokens, self.inspection_void.expand(batch, 1, -1)), dim=1)
        region_tokens = query_tokens[:, None] + pooled + self.inspection_geometry(geometry)
        region_tokens = region_tokens[images, active_slots]
        area = area[images, active_slots]
        valid = area > 0
        capacity = int(valid.sum(-1).max())
        slots = valid.long().argsort(dim=-1, descending=True, stable=True)[..., :capacity]
        region_tokens = region_tokens.gather(1, slots[..., None].expand(-1, -1, region_tokens.shape[-1]))
        area = area.gather(1, slots)
        tokens = torch.cat((self.inspection_cls.expand(len(region_tokens), 1, -1), region_tokens), dim=1)
        padding = F.pad(area == 0, (1, 0), value=False)

        def encode(tokens):
            with torch.autocast(device_type=tokens.device.type,
                    enabled=torch.is_autocast_enabled(tokens.device.type), cache_enabled=False):
                return self.inspection_encoder(tokens, src_key_padding_mask=padding)[:, 0]

        encoded = (checkpoint(encode, tokens, use_reentrant=False)
                   if self.training and torch.is_grad_enabled() else encode(tokens))
        if memo is not None:
            encoded = memo.store_unique(keys, encoded) if unique_states else memo.store(lookup, encoded)
        return encoded.reshape(batch, states, -1)
