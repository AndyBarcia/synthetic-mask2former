"""Map-conditioned pointer decoder over mask query embeddings."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .map_encoder import MapEncoder
from .state_cache import StateCache, pack_by_image
from .mask_groups import mask_compatibility, group_logits, SampledTrajectory


class ObjectDecoder(nn.Module):
    def __init__(self, mask_dim, hidden_dim, num_queries, num_layers, num_heads,
                 dim_feedforward, mask_iou_threshold=0.8, prefix_tree_branches=4,
                 eof_threshold=.5,
                 collapse_masks=False, collapse_iou_threshold=.8, map_feature_dim=None):
        super().__init__()
        if not 0 <= mask_iou_threshold <= 1:
            raise ValueError("mask_iou_threshold must be between 0 and 1")
        if prefix_tree_branches < 1:
            raise ValueError("prefix_tree_branches must be positive")
        self.prefix_tree_branches = prefix_tree_branches
        self.num_queries = num_queries
        self.num_heads = num_heads
        if not 0 <= eof_threshold < 1:
            raise ValueError("EOF threshold must be in [0, 1); zero restores argmax")
        if not 0 < collapse_iou_threshold < 1:
            raise ValueError("Collapse IoU threshold must be in (0, 1)")
        self.collapse_masks = collapse_masks
        self.collapse_iou_threshold = collapse_iou_threshold
        self.eof_threshold = eof_threshold
        self.mask_iou_threshold = mask_iou_threshold
        self.input_mask_projection = nn.Linear(mask_dim, hidden_dim)
        self.output_mask_projection = nn.Linear(mask_dim, hidden_dim)
        self.eof = nn.Parameter(torch.zeros(hidden_dim))
        # This key remains available if previous masks cover an entire feature map.
        self.null_image_token = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.normal_(self.null_image_token, std=0.02)
        self.map_encoder = MapEncoder(hidden_dim, num_heads, dim_feedforward, num_queries, map_feature_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=0.0, batch_first=True,
        )
        self.layers = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.vocabulary_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
            for _ in range(num_layers)
        ])
        self.vocabulary_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = hidden_dim ** -0.5

    @staticmethod
    def pack_prefix_trees(sequences, num_queries, device, path_weights=None):
        """Pack paths into context tries; edge counts preserve expanded-path CE.

        Each node represents a prefix (root is BOS). Targets are outgoing edge
        labels, including EOF. Shared nodes accumulate one count per path.
        Padding attends only to itself and has no loss weight.
        """
        trees = []
        for image, paths in enumerate(sequences):
            tokens, parents, depths, counts = [-1], [-1], [0], [{}]
            children = {}
            for path_index, path in enumerate(paths):
                path_weight = 1.0 if path_weights is None else float(path_weights[image][path_index])
                node = 0
                for token in list(path) + [num_queries]:
                    token = int(token)
                    counts[node][token] = counts[node].get(token, 0) + path_weight
                    if token == num_queries:
                        break
                    key = (node, token)
                    if key not in children:
                        children[key] = len(tokens)
                        tokens.append(token)
                        parents.append(node)
                        depths.append(depths[node] + 1)
                        counts.append({})
                    node = children[key]
            trees.append((tokens, parents, depths, counts))
        size = max(len(tree[0]) for tree in trees)
        previous = torch.full((len(trees), size), -1, device=device, dtype=torch.long)
        positions = torch.zeros_like(previous)
        ancestors = torch.eye(size, device=device, dtype=torch.bool).expand(len(trees), -1, -1).clone()
        weights = torch.zeros(len(trees), size, num_queries + 1, device=device)
        for batch, (tokens, parents, depths, counts) in enumerate(trees):
            previous[batch, :len(tokens)] = torch.tensor(tokens, device=device)
            positions[batch, :len(tokens)] = torch.tensor(depths, device=device)
            for node, parent in enumerate(parents):
                if parent >= 0:
                    ancestors[batch, node] |= ancestors[batch, parent]
                for token, count in counts[node].items():
                    weights[batch, node, token] = count
        return previous, positions, ancestors, weights

    @staticmethod
    def _accumulate_prior(prior, valid, ancestors=None):
        prior = prior & valid.unsqueeze(-1)
        if ancestors is None:
            return prior.cumsum(dim=1) > 0
        return torch.bmm(ancestors.float(), prior.float()) > 0

    @staticmethod
    def prepare_regions(mask_logits, image_sizes):
        """Threshold final query masks at each cross-attention feature scale."""
        with torch.no_grad():
            return [
                (F.interpolate(mask_logits.float(), size=size, mode="bilinear", align_corners=False)
                 .flatten(2) > 0)
                for size in image_sizes
            ]

    def prepare_vocabulary_exclusions(self, mask_logits):
        """Find masks to exclude after selecting each query, including the query itself."""
        with torch.no_grad():
            masks = (mask_logits.detach().flatten(2) > 0).float()
            intersection = torch.bmm(masks, masks.transpose(1, 2))
            area = masks.sum(-1)
            union = area[:, :, None] + area[:, None, :] - intersection
            exclusions = intersection / union.clamp_min(1) > self.mask_iou_threshold
            identity = torch.eye(masks.shape[1], dtype=torch.bool, device=masks.device)
            return exclusions | identity

    @staticmethod
    def _vocabulary_mask(exclusions, previous_tokens, ancestors=None):
        """Exclude selected masks and their IoU neighbors from later positions."""
        batch_size, num_queries, _ = exclusions.shape
        indices = previous_tokens.clamp(0, num_queries - 1)
        prior = exclusions.gather(1, indices.unsqueeze(-1).expand(-1, -1, num_queries))
        valid = (previous_tokens >= 0) & (previous_tokens < num_queries)
        blocked = ObjectDecoder._accumulate_prior(prior, valid, ancestors)
        # EOF remains available even when every mask has been excluded.
        return F.pad(blocked, (0, 1), value=False)

    def _memory_mask(self, regions, previous_tokens, ancestors=None):
        """Block the union of masks emitted up to each map prefix."""
        batch_size, num_queries, spatial_tokens = regions.shape
        indices = previous_tokens.clamp(0, num_queries - 1)
        prior_regions = regions.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, spatial_tokens)
        )
        valid = (previous_tokens >= 0) & (previous_tokens < num_queries)
        covered = self._accumulate_prior(prior_regions, valid, ancestors)
        # The learned null image key must never be masked, including at full coverage.
        covered = F.pad(covered, (0, 1), value=False)
        return covered[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            batch_size * self.num_heads, previous_tokens.shape[1], spatial_tokens + 1
        )

    def forward(self, mask_embeddings, image_features, image_regions, previous_tokens,
                vocabulary_exclusions, *, positions=None, ancestors=None, return_hidden=False,
                map_last_only=False, state_cache=None, image_ids=None, merge_states=True):
        # Sampling and gradient scoring can share an outer AMP context.
        # Avoid reusing parameter casts cached by the preceding no-grad pass.
        with torch.autocast(device_type=mask_embeddings.device.type,
                enabled=torch.is_autocast_enabled(mask_embeddings.device.type), cache_enabled=False):
            return self._forward_impl(mask_embeddings, image_features, image_regions,
                previous_tokens, vocabulary_exclusions, positions=positions,
                ancestors=ancestors, return_hidden=return_hidden, map_last_only=map_last_only,
                state_cache=state_cache, image_ids=image_ids, merge_states=merge_states)

    def _forward_impl(self, mask_embeddings, image_features, image_regions, previous_tokens,
                      vocabulary_exclusions, *, positions=None, ancestors=None,
                      return_hidden=False, map_last_only=False, state_cache=None,
                      image_ids=None, merge_states=True):
        cache = self._prepare_decode_cache(mask_embeddings, image_features, image_regions,
            vocabulary_exclusions, previous_tokens.shape[1], state_cache=state_cache,
            image_ids=image_ids, merge_states=merge_states)
        owner = self.map_encoder.render_maps(cache["map_regions"], previous_tokens,
            ancestors, positions, map_last_only)
        valid = (previous_tokens >= 0) & (previous_tokens < self.num_queries)
        prior = F.one_hot(previous_tokens.clamp(0, self.num_queries - 1), self.num_queries).bool()
        selected = self._accumulate_prior(prior, valid, ancestors)
        blocked = self._vocabulary_mask(vocabulary_exclusions, previous_tokens, ancestors)
        if map_last_only:
            selected, blocked = selected[:, -1:], blocked[:, -1:]
        self._prove_unique_path(previous_tokens, selected, cache["map_regions"], cache,
                                ancestors, positions)
        covered = [F.pad(torch.bmm(selected.float(), region.float()) > 0, (0, 1))
                   for region in image_regions[:3]]
        logits, hidden = self._evaluate_states(owner, selected, covered, blocked, cache)
        if map_last_only:
            logits = F.pad(logits, (0, 0, previous_tokens.shape[1] - 1, 0))
            hidden = F.pad(hidden, (0, 0, previous_tokens.shape[1] - 1, 0))
        return (logits, hidden) if return_hidden else logits

    @staticmethod
    def _prove_unique_path(previous, selected, regions, cache, ancestors, positions):
        """Exact cold-cache shortcut for distinct linear prefixes.

        A newly selected nonempty query paints a label absent from every prior
        map. Selected sets themselves remain distinct even for empty masks.
        Independent source images cannot collide with each other.
        """
        scope = cache["state_cache"]
        if (not cache["merge_states"] or ancestors is not None or positions is not None
                or len(cache["memory_rows"]) != len(previous)
                or scope["maps"].keys is not None or scope["decoder"].keys is not None):
            return
        if selected.shape[1] == 1:
            cache["unique_maps"] = cache["unique_decoder_states"] = True
            return
        introduced = selected[:, 1:] & ~selected[:, :-1]
        fresh = introduced.any(-1).all()
        nonempty = (introduced & regions.any(-1)[:, None]).any(-1).all()
        # One small transfer replaces two full-key sorts and compaction passes.
        cache["unique_decoder_states"], cache["unique_maps"] = torch.stack((fresh, nonempty)).tolist()

    def _evaluate_states(self, owner, selected, covered, blocked, cache):
        with torch.autocast(device_type=owner.device.type,
                enabled=torch.is_autocast_enabled(owner.device.type), cache_enabled=False):
            return self._evaluate_states_impl(owner, selected, covered, blocked, cache)

    def _evaluate_states_impl(self, owner, selected, covered, blocked, cache):
        batch, steps, _ = owner.shape
        scope = cache["state_cache"]
        summaries = self.map_encoder.encode_maps(cache["vocabulary"][:, :-1],
            cache["map_features"], cache["map_regions"], owner, cache["pool_cache"],
            state_cache=scope["maps"], image_ids=cache["image_ids"],
            merge_states=cache["merge_states"], unique_states=cache["unique_maps"],
            inference_capacity=cache.get("inference_capacity"))
        memo = scope["decoder"] if cache["merge_states"] else None
        if memo is not None:
            # Selected queries determine the image exclusions at *every* scale.
            # Blocked vocabulary is explicit so equal visible maps cannot erase
            # action eligibility differences (e.g. fully overwritten queries).
            keys = torch.cat((cache["image_ids"][:, None, None].expand(-1, steps, 1),
                owner, selected, blocked), dim=-1).flatten(0, 1)
            if cache["unique_decoder_states"]:
                missing = torch.arange(batch * steps, device=owner.device)
            else:
                lookup = memo.lookup(keys)
                missing = lookup.missing
        else:
            missing = torch.arange(batch * steps, device=owner.device)
        if len(missing):
            memory_batch = len(cache["memory_rows"])
            dense = (memo is None or cache["unique_decoder_states"]) and memory_batch == batch
            if dense:
                # Proven-unique queries already have their dense [image, state]
                # layout. No sorting, packing, scatter, or result gather needed.
                if cache["identity_images"]:
                    decoded, memory_masks, vocabulary_mask = summaries, covered, blocked
                else:
                    rows = cache["memory_rows"]
                    decoded = summaries[rows]
                    memory_masks = [mask[rows] for mask in covered]
                    vocabulary_mask = blocked[rows]
            else:
                images, nodes = missing // steps, missing % steps
                memory_images = cache["memory_inverse"][images]
                slots, capacity = pack_by_image(memory_images, memory_batch)
                decoded = summaries.new_zeros(memory_batch, capacity, summaries.shape[-1])
                decoded[memory_images, slots] = summaries[images, nodes].to(decoded.dtype)
                memory_masks = []
                for mask in covered:
                    packed = mask.new_zeros(memory_batch, capacity, mask.shape[-1])
                    packed[memory_images, slots] = mask[images, nodes]
                    memory_masks.append(packed)
                vocabulary_mask = blocked.new_zeros(memory_batch, capacity, blocked.shape[-1])
                vocabulary_mask[memory_images, slots] = blocked[images, nodes]
            for index, layer in enumerate(self.layers.layers):
                # Each map has exactly one self-attention key. Its softmax is
                # identically one: only the value and output projections remain.
                width = layer.self_attn.embed_dim
                attended = F.linear(decoded, layer.self_attn.in_proj_weight[2 * width:],
                                    layer.self_attn.in_proj_bias[2 * width:])
                attended = layer.self_attn.out_proj(attended)
                decoded = layer.norm1(decoded + layer.dropout1(attended))
                attended = self._cached_cross_attention(layer.multihead_attn, decoded,
                    cache["image_kv"][index], memory_masks[index % 3])
                decoded = layer.norm2(decoded + layer.dropout2(attended))
                decoded = layer.norm3(decoded + layer._ff_block(decoded))
                attended = self._cached_cross_attention(self.vocabulary_attention[index], decoded,
                    cache["vocabulary_kv"][index], vocabulary_mask)
                decoded = self.vocabulary_norm[index](decoded + attended)
            hidden = self.norm(decoded)
            logits = (hidden @ cache["output_vocabulary"].transpose(1, 2)) * self.scale
            if dense:
                values = torch.cat((logits, hidden), -1)
                if not cache["identity_images"]:
                    values = values[cache["memory_inverse"]]
                values = values.flatten(0, 1)
            else:
                values = torch.cat((logits[memory_images, slots], hidden[memory_images, slots]), -1)
        else:
            values = None
        if memo is not None:
            values = (memo.store_unique(keys, values) if cache["unique_decoder_states"]
                      else memo.store(lookup, values))
        values = values.reshape(batch, steps, -1)
        logits_dtype = (torch.get_autocast_dtype(owner.device.type)
                        if torch.is_autocast_enabled(owner.device.type)
                        else cache["output_vocabulary"].dtype)
        return values[..., :self.num_queries + 1].to(logits_dtype), values[..., self.num_queries + 1:]

    def _state_scope(self, state_cache):
        scope = {} if state_cache is None else state_cache
        precision = (torch.get_autocast_dtype(self.eof.device.type)
                     if torch.is_autocast_enabled(self.eof.device.type) else None)
        signature = (id(self), torch.is_grad_enabled(), self.training, self.eof.device,
                     self.eof.dtype, precision, tuple(p._version for p in self.parameters()))
        if "signature" in scope and scope["signature"] != signature:
            raise ValueError("State caches must not cross models, updates, grad modes, or precision contexts")
        scope["signature"] = signature
        if "maps" not in scope:
            scope["maps"], scope["decoder"] = StateCache(), StateCache()
        return scope

    @staticmethod
    def _attention_heads(tensor, num_heads):
        batch, steps, width = tensor.shape
        return tensor.reshape(batch, steps, num_heads, width // num_heads).transpose(1, 2)

    def _project_attention_kv(self, attention, memory):
        width = attention.embed_dim
        bias = attention.in_proj_bias
        key, value = F.linear(memory, attention.in_proj_weight[width:],
                              bias[width:] if bias is not None else None).chunk(2, dim=-1)
        return (self._attention_heads(key, attention.num_heads),
                self._attention_heads(value, attention.num_heads))

    def _cached_cross_attention(self, attention, query, kv, blocked):
        width = attention.embed_dim
        bias = attention.in_proj_bias
        query = F.linear(query, attention.in_proj_weight[:width],
                         bias[:width] if bias is not None else None)
        query = self._attention_heads(query, attention.num_heads)
        # SDPA uses True for allowed keys, unlike MultiheadAttention.
        attended = F.scaled_dot_product_attention(
            query, *kv, attn_mask=(~blocked)[:, None, :, :],
        )
        attended = attended.transpose(1, 2).reshape(query.shape[0], query.shape[2], width)
        return attention.out_proj(attended)

    def _fixed_memories(self, mask_embeddings, image_features, rows, image_ids, scope):
        """Source-image K/V projections shared across rollout/scoring batches."""
        if "memory_ids" in scope and torch.equal(scope["memory_ids"], image_ids):
            return scope["memories"]
        batch = len(rows)
        vocabulary = torch.cat((self.input_mask_projection(mask_embeddings[rows]),
                                self.eof.expand(batch, 1, -1)), 1)
        output = torch.cat((self.output_mask_projection(mask_embeddings[rows]),
                            self.eof.expand(batch, 1, -1)), 1)
        images = [torch.cat((feature[rows], self.null_image_token[level].expand(batch, 1, -1)), 1)
                  for level, feature in enumerate(image_features[:3])]
        memories = {
            "vocabulary": vocabulary,
            "output_vocabulary": output,
            "image_kv": [self._project_attention_kv(layer.multihead_attn, images[index % 3])
                         for index, layer in enumerate(self.layers.layers)],
            "vocabulary_kv": [self._project_attention_kv(attention, vocabulary)
                              for attention in self.vocabulary_attention],
        }
        scope["memory_ids"], scope["memories"] = image_ids.clone(), memories
        return memories

    def _prepare_decode_cache(self, mask_embeddings, image_features, regions,
                              exclusions, max_steps, *, state_cache=None, image_ids=None,
                              merge_states=True):
        """Cache fixed projections and allocate incremental, rollout-local state.

        Projected tensors retain their gradients during differentiable rollouts.
        State memos can span rollout batches in one explicitly shared scope.
        The scope must not be reused after backward or an optimizer update.
        """
        if len(image_features) not in (3, 4) or len(regions) != len(image_features):
            raise ValueError("Expected three attention levels and an optional full-resolution map level")
        map_level = 3 if len(image_features) == 4 else 0
        batch_size = mask_embeddings.shape[0]
        private_scope = state_cache is None
        state_cache = self._state_scope(state_cache)
        identity_images = image_ids is None
        if image_ids is None:
            image_ids = torch.arange(batch_size, device=mask_embeddings.device)
            memory_ids = memory_rows = memory_inverse = image_ids
        else:
            image_ids = image_ids.to(device=mask_embeddings.device, dtype=torch.long)
            if image_ids.shape != (batch_size,):
                raise ValueError("image_ids must contain one stable source ID per batch row")
            memory_ids, memory_inverse = torch.unique(image_ids, sorted=True, return_inverse=True)
            memory_rows = torch.full_like(memory_ids, batch_size)
            memory_rows.scatter_reduce_(0, memory_inverse,
                torch.arange(batch_size, device=image_ids.device), reduce="amin")
        with torch.autocast(device_type=mask_embeddings.device.type,
                enabled=torch.is_autocast_enabled(mask_embeddings.device.type), cache_enabled=False):
            memories = self._fixed_memories(mask_embeddings, image_features, memory_rows,
                                            memory_ids, state_cache)
            map_features = self.map_encoder.feature_projection(image_features[map_level])
        vocabulary = memories["vocabulary"][memory_inverse]
        return {
            "vocabulary": vocabulary,
            "output_vocabulary": memories["output_vocabulary"],
            "image_kv": memories["image_kv"],
            "vocabulary_kv": memories["vocabulary_kv"],
            "memory_rows": memory_rows,
            "memory_inverse": memory_inverse,
            "features": image_features,
            "map_features": map_features,
            "map_regions": regions[map_level],
            "pool_cache": {"float32_accumulation": map_level == 3},
            "state_cache": state_cache,
            "image_ids": image_ids,
            "merge_states": merge_states,
            "unique_maps": False,
            "unique_decoder_states": False,
            "identity_images": identity_images,
            "private_scope": private_scope,
            "owner": torch.full((batch_size, regions[map_level].shape[-1]), self.num_queries,
                                dtype=torch.long, device=mask_embeddings.device),
            "selected": exclusions.new_zeros(batch_size, self.num_queries),
            "regions": regions,
            "exclusions": exclusions,
            "covered": [region.new_zeros(batch_size, region.shape[-1] + 1) for region in regions[:3]],
            "blocked": exclusions.new_zeros(batch_size, self.num_queries + 1),
            "max_steps": max_steps,
            "step": 0,
        }

    def _decode_step(self, previous_token, cache):
        """Consume one token (BOS=-1), returning logits for the next action."""
        step = cache["step"]
        if step >= cache["max_steps"]:
            raise ValueError("Decode cache capacity exceeded")
        vocabulary = cache["vocabulary"]
        batch_size = vocabulary.shape[0]
        rows = torch.arange(batch_size, device=previous_token.device)
        valid = (previous_token >= 0) & (previous_token < self.num_queries)
        query_index = previous_token.clamp(0, self.num_queries - 1)
        cache["blocked"][:, :-1] |= cache["exclusions"][rows, query_index] & valid[:, None]
        for covered, regions in zip(cache["covered"], cache["regions"]):
            covered[:, :-1] |= regions[rows, query_index] & valid[:, None]
        paint = cache["map_regions"][rows, query_index] & valid[:, None]
        cache["unique_maps"] = cache["unique_decoder_states"] = False
        if (cache["merge_states"] and cache["private_scope"]
                and len(cache["memory_rows"]) == batch_size):
            scope = cache["state_cache"]
            if scope["maps"].keys is None and scope["decoder"].keys is None:
                cache["unique_maps"] = cache["unique_decoder_states"] = True
            else:
                fresh = valid & ~cache["selected"][rows, query_index]
                flags = torch.stack((fresh.all(), (fresh & paint.any(-1)).all())).tolist()
                cache["unique_decoder_states"], cache["unique_maps"] = flags
        cache["owner"] = torch.where(paint, query_index[:, None], cache["owner"])
        cache["selected"] = cache["selected"] | (F.one_hot(query_index, self.num_queries).bool() & valid[:, None])
        logits, _ = self._evaluate_states(cache["owner"][:, None], cache["selected"][:, None],
            [mask[:, None].clone() for mask in cache["covered"]], cache["blocked"][:, None].clone(), cache)
        cache["step"] += 1
        return logits[:, 0].clone()

    def _rollout_cache(self, use_cache, mask_embeddings, image_features, regions,
                       exclusions, max_steps, *, state_cache=None, image_ids=None):
        # Respect custom forward implementations.
        if (use_cache
                and getattr(self.forward, "__func__", None) is ObjectDecoder.forward):
            return self._prepare_decode_cache(mask_embeddings, image_features, regions,
                                              exclusions, max_steps, state_cache=state_cache, image_ids=image_ids)
        return None

    def _greedy_action(self, logits, threshold=None):
        """Strict majority EOF for greedy decoding; sampling stays categorical."""
        threshold = self.eof_threshold if threshold is None else threshold
        if not 0 <= threshold < 1:
            raise ValueError("EOF threshold must be in [0, 1)")
        if threshold == 0:
            return logits.argmax(-1)
        stop = logits.float().softmax(-1)[..., -1] > threshold
        mask = logits[..., :-1].argmax(-1)
        return torch.where(stop, torch.full_like(mask, self.num_queries), mask)

    @torch.no_grad()
    def generate(self, mask_embeddings, image_features, mask_logits, image_sizes, *, use_cache=True, image_regions=None, query_classes=None):
        """Return ordered queries, mask-versus-EOF logits, and probabilities."""
        batch_size, num_queries, _ = mask_embeddings.shape
        if image_regions is None:
            image_regions = self.prepare_regions(mask_logits, image_sizes)
        exclusions = self.prepare_vocabulary_exclusions(mask_logits)
        compatibility = (mask_compatibility(mask_logits, query_classes, self.collapse_iou_threshold)
                         if self.collapse_masks else None)
        unavailable = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=mask_embeddings.device)
        order = torch.full((batch_size, num_queries + 1), -1, dtype=torch.long, device=mask_embeddings.device)
        log_odds = torch.zeros(batch_size, num_queries + 1, dtype=mask_embeddings.dtype, device=mask_embeddings.device)
        probabilities = torch.zeros(batch_size, num_queries + 1, dtype=mask_embeddings.dtype, device=mask_embeddings.device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=mask_embeddings.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=mask_embeddings.device)
        tokens = torch.full((batch_size, 1), -1, dtype=torch.long, device=mask_embeddings.device)
        rows = torch.arange(batch_size, device=mask_embeddings.device)
        cache = self._rollout_cache(use_cache, mask_embeddings, image_features, image_regions, exclusions, num_queries)
        if cache is not None:
            # A private greedy path has no cross-rollout states to reuse. Keep
            # fixed K/V and pooled segments, without accumulating state history.
            cache["merge_states"] = False
        for step in range(num_queries):
            if cache is not None:
                cache["inference_capacity"] = min(step + 1, num_queries + 1)
            logits = (self._decode_step(tokens[:, -1], cache) if cache is not None else
                      self(mask_embeddings, image_features, image_regions, tokens, exclusions,
                           map_last_only=True)[:, -1])
            logits[:, :num_queries].masked_fill_(unavailable, -torch.inf)
            selection_logits = group_logits(logits.float(), compatibility) if compatibility is not None else logits
            next_token = self._greedy_action(selection_logits)
            active = ~finished & (next_token != num_queries)
            query = next_token.clamp_max(num_queries - 1)
            unavailable |= exclusions[rows, query] & active[:, None]
            # Finished rows write a sentinel column instead of packing a
            # variable-length active batch (which synchronizes CUDA).
            destination = torch.where(active, lengths, num_queries)
            order[rows, destination] = next_token
            relative_logits = selection_logits[rows, query] - selection_logits[rows, num_queries]
            log_odds[rows, destination] = relative_logits.to(log_odds.dtype)
            probabilities[rows, destination] = relative_logits.sigmoid().to(probabilities.dtype)
            lengths += active.long()
            finished |= next_token == num_queries
            if finished.all():
                break
            next_input = torch.where(finished, num_queries, next_token)[:, None]
            tokens = next_input if cache is not None else torch.cat((tokens, next_input), dim=1)
        return [
            (order[batch_index, :lengths[batch_index]],
             log_odds[batch_index, :lengths[batch_index]],
             probabilities[batch_index, :lengths[batch_index]])
            for batch_index in range(batch_size)
        ]

    @torch.no_grad()
    def rollout_samples_and_greedy(self, mask_embeddings, image_features, mask_logits,
                                   image_sizes, num_samples, *, max_steps=32,
                                   query_classes=None, query_bias_logits=None,
                                   state_cache=None):
        """Decode all samples and the greedy baseline in a shared state batch.

        Copies are sample-major, with the greedy batch last. Fixed proposal
        regions, IoU exclusions, and complete-link compatibility are prepared
        once per source image. Sampling and differentiable rescoring must use
        separate state scopes.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        batch = mask_embeddings.shape[0]
        repeats = num_samples + 1
        regions = [x.repeat(repeats, 1, 1) for x in self.prepare_regions(mask_logits, image_sizes)]
        exclusions = self.prepare_vocabulary_exclusions(mask_logits).repeat(repeats, 1, 1)
        compatibility = (mask_compatibility(mask_logits, query_classes, self.collapse_iou_threshold)
                         if self.collapse_masks else None)
        if compatibility is not None:
            compatibility = compatibility[None].repeat(repeats, axis=0).reshape(
                repeats * batch, self.num_queries, self.num_queries)
        orders, scores, actions = self.rollout(
            mask_embeddings.repeat(repeats, 1, 1),
            [x.repeat(repeats, 1, 1) for x in image_features],
            None, image_sizes,  # All mask-derived data is supplied in _prepared.
            max_steps=max_steps, return_actions=True, sample_rows=num_samples * batch,
            query_bias_logits=(query_bias_logits.repeat(repeats, 1)
                               if query_bias_logits is not None else None),
            image_ids=torch.arange(batch, device=mask_embeddings.device).repeat(repeats),
            state_cache={} if state_cache is None else state_cache,
            _prepared=(regions, exclusions, compatibility))
        return orders[:-batch], orders[-batch:], actions[:-batch], actions[-batch:]

    def rollout(self, mask_embeddings, image_features, mask_logits, image_sizes,
                sample=True, max_steps=32, return_diagnostics=False,
                return_actions=False, query_bias_logits=None, *, use_cache=True, greedy_eof_threshold=None,
                query_classes=None, state_cache=None, image_ids=None,
                sample_rows=None, _prepared=None):
        """Generate query orders and sequence log probabilities for policy gradients.

        A capped rollout is treated as an implicit EOF after the last selected query.
        """
        batch_size, num_queries = mask_embeddings.shape[:2]
        max_steps = min(max_steps, num_queries + 1)
        if _prepared is None:
            regions = self.prepare_regions(mask_logits, image_sizes)
            exclusions = self.prepare_vocabulary_exclusions(mask_logits)
            compatibility = (mask_compatibility(mask_logits, query_classes, self.collapse_iou_threshold)
                             if self.collapse_masks else None)
        else:
            regions, exclusions, compatibility = _prepared
        sampled_count = (batch_size if sample else 0) if sample_rows is None else sample_rows
        if not 0 <= sampled_count <= batch_size:
            raise ValueError("sample_rows must be within the rollout batch")
        sampled_rows = torch.arange(batch_size, device=mask_embeddings.device) < sampled_count
        unavailable = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=mask_embeddings.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=mask_embeddings.device)
        previous = torch.full((batch_size, 1), -1, dtype=torch.long, device=mask_embeddings.device)
        action_buffer = torch.full((batch_size, max_steps), -1, dtype=torch.long, device=mask_embeddings.device)
        sampled_groups = []
        rows = torch.arange(batch_size, device=mask_embeddings.device)
        log_probability = mask_embeddings.new_zeros(batch_size)
        diagnostics = {
            key: torch.zeros(batch_size, device=mask_embeddings.device)
            for key in ("actions", "entropy", "top1_probability", "eof_probability", "available_masks")
        } if return_diagnostics else None
        cache = self._rollout_cache(use_cache, mask_embeddings, image_features, regions, exclusions, max_steps,
                                    state_cache=state_cache, image_ids=image_ids)
        for step in range(max_steps):
            if cache is not None and not torch.is_grad_enabled():
                cache["inference_capacity"] = min(step + 1, num_queries + 1)
            logits = (self._decode_step(previous[:, -1], cache) if cache is not None else
                      self(mask_embeddings, image_features, regions, previous, exclusions,
                           map_last_only=True, state_cache=state_cache, image_ids=image_ids)[:, -1]).clone()
            logits[:, :num_queries].masked_fill_(unavailable | finished[:, None], -torch.inf)
            # Finished paths have no policy contribution. A degenerate EOF
            # distribution avoids regrouping their unused vocabulary forever.
            logits[:, -1].masked_fill_(finished, 0)
            if compatibility is not None:
                if return_actions:
                    logits, groups = group_logits(logits.float(), compatibility, return_indices=True)
                    sampled_groups.append(groups)
                else:
                    logits = group_logits(logits.float(), compatibility)
            distribution = torch.distributions.Categorical(logits=logits)
            if sampled_count == batch_size:
                token = distribution.sample()
            elif sampled_count == 0:
                token = self._greedy_action(logits, greedy_eof_threshold)
            else:
                # Greedy rows consume no random draws and retain the strict EOF gate.
                token = torch.cat((torch.distributions.Categorical(logits=logits[:sampled_count]).sample(),
                    self._greedy_action(logits[sampled_count:], greedy_eof_threshold)))
            active = ~finished
            if diagnostics is not None:
                with torch.no_grad():
                    probabilities = distribution.probs.detach()
                    diagnostics["actions"] += active.float()
                    diagnostics["entropy"] += torch.where(active, distribution.entropy().detach(), 0)
                    diagnostics["top1_probability"] += torch.where(active, probabilities.max(-1).values, 0)
                    diagnostics["eof_probability"] += torch.where(active, probabilities[:, -1], 0)
                    diagnostics["available_masks"] += torch.where(active, (~unavailable).sum(-1), 0)
            if sampled_count:
                log_probability = log_probability + torch.where(
                    active & sampled_rows, distribution.log_prob(token), 0.0
                )
            action_buffer[:, step] = torch.where(active, token, -1).detach()
            chosen = active & (token != num_queries)
            unavailable = unavailable | (exclusions[rows, token.clamp_max(num_queries - 1)] & chosen[:, None])
            finished = finished | (token == num_queries)
            if finished.all():
                break
            next_input = torch.where(finished, num_queries, token)[:, None]
            previous = next_input if cache is not None else torch.cat((previous, next_input), dim=1)
        # Transfer the complete ragged trajectories once, rather than each token.
        rows = action_buffer.tolist()
        actions = [[token for token in row if token >= 0] for row in rows]
        if sampled_groups:
            groups = np.stack(sampled_groups, axis=1)
            actions = [SampledTrajectory(row, groups[index, :len(row)].copy())
                       for index, row in enumerate(actions)]
        orders = [[token for token in row if token != num_queries] for row in actions]
        if diagnostics is not None:
            diagnostics["length"] = torch.tensor([len(order) for order in orders], device=finished.device).float()
            diagnostics["truncated"] = (~finished).float()
            if return_actions:
                return orders, log_probability, diagnostics, actions
            return orders, log_probability, diagnostics
        if return_actions:
            return orders, log_probability, actions
        return orders, log_probability

    def trajectory_log_probability(self, mask_embeddings, image_features,
                                       mask_logits, image_sizes, trajectories,
                                       query_bias_logits=None, *, query_classes=None,
                                       state_cache=None, image_ids=None):
        """Score sampled mask and EOF actions from their prefix maps."""
        batch_size, num_queries = mask_embeddings.shape[:2]
        steps = max(map(len, trajectories))
        actions = torch.full(
            (batch_size, steps), num_queries, dtype=torch.long, device=mask_embeddings.device
        )
        lengths = torch.tensor(
            [len(trajectory) for trajectory in trajectories], device=actions.device
        )
        for index, trajectory in enumerate(trajectories):
            if not trajectory or trajectory[-1] != num_queries:
                raise ValueError("Every trajectory must end with an EOF action")
            actions[index, :len(trajectory)] = torch.as_tensor(trajectory, device=actions.device)
        previous = torch.cat((
            torch.full((batch_size, 1), -1, dtype=torch.long, device=actions.device),
            actions[:, :-1],
        ), dim=1)
        exclusions = self.prepare_vocabulary_exclusions(mask_logits)
        unavailable = torch.zeros(
            batch_size, num_queries, dtype=torch.bool, device=actions.device
        )
        allowed = []
        for step in range(steps):
            active = lengths > step
            allowed.append(torch.cat((~unavailable & active[:, None],
                                      torch.ones(batch_size, 1, dtype=torch.bool,
                                                 device=actions.device)), dim=-1))
            chosen = active & (actions[:, step] != num_queries)
            chosen_batch = chosen.nonzero(as_tuple=True)[0]
            unavailable[chosen_batch] |= exclusions[chosen_batch, actions[chosen_batch, step]]
        regions = self.prepare_regions(mask_logits, image_sizes)
        logits = self(mask_embeddings, image_features, regions, previous, exclusions,
                      state_cache=state_cache, image_ids=image_ids)
        logits = logits.float().masked_fill(~torch.stack(allowed, dim=1), -torch.inf)
        if self.collapse_masks:
            # Cached incremental decoding and batched rescoring can round nearly
            # tied logits differently. Replay the sampled complete-link topology,
            # including its representative, so the sampled event retains support.
            saved_groups = [getattr(path, "group_indices", None) for path in trajectories]
            group_indices = None
            if any(group is not None for group in saved_groups):
                if not all(group is not None for group in saved_groups):
                    raise ValueError("Cannot mix grouped sampled and plain trajectories")
                group_indices = np.broadcast_to(
                    np.arange(num_queries + 1), (batch_size, steps, num_queries + 1)).copy()
                for index, group in enumerate(saved_groups):
                    group_indices[index, :len(group)] = group
            logits = group_logits(logits, mask_compatibility(
                mask_logits, query_classes, self.collapse_iou_threshold),
                group_indices=group_indices)
        log_probabilities = F.log_softmax(logits, dim=-1).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        return torch.where(
            torch.arange(steps, device=actions.device)[None, :] < lengths[:, None],
            log_probabilities, 0.0,
        ).sum(-1)

    def set_imitation_loss(self, mask_embeddings, image_features, mask_logits,
                           image_sizes, target_sets, *, use_cache=True):
        """Imitate an unordered query set using constrained greedy roll-in.

        The roll-in chooses its own order among target queries under no_grad.
        At each prefix, the supervised event is *any* remaining target query;
        after the set is complete, the supervised event is EOF.
        """
        batch_size, num_queries = mask_embeddings.shape[:2]
        if len(target_sets) != batch_size:
            raise ValueError("target_sets must contain one set per image")
        device = mask_embeddings.device
        target = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=device)
        for image, queries in enumerate(target_sets):
            if len(set(queries)) != len(queries) or any(
                query < 0 or query >= num_queries for query in queries
            ):
                raise ValueError("target queries must be unique and in range")
            target[image, queries] = True
        exclusions = self.prepare_vocabulary_exclusions(mask_logits)
        regions = self.prepare_regions(mask_logits, image_sizes)

        # Choose prefixes from the current policy, constrained to the target
        # set. The sampled best trajectory contributes only its selected set.
        with torch.no_grad():
            remaining = target.clone()
            unavailable = torch.zeros_like(target)
            previous = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
            rollin = [[] for _ in range(batch_size)]
            max_steps = int(target.sum(-1).max().item())
            cache = self._rollout_cache(use_cache, mask_embeddings, image_features, regions, exclusions, max_steps)
            for _ in range(max_steps):
                active = remaining.any(-1)
                eligible = remaining & ~unavailable
                if (active & ~eligible.any(-1)).any():
                    raise ValueError("Target set contains mutually excluded queries")
                logits = (self._decode_step(previous[:, -1], cache) if cache is not None else
                          self(mask_embeddings, image_features, regions, previous, exclusions)[:, -1]).float()
                logits[:, :num_queries].masked_fill_(~eligible, -torch.inf)
                logits[active, num_queries] = -torch.inf
                logits[~active, num_queries] = 0
                chosen = logits.argmax(-1)
                for image in range(batch_size):
                    if active[image]:
                        rollin[image].append(int(chosen[image]))
                active_rows = active.nonzero(as_tuple=True)[0]
                remaining[active_rows, chosen[active_rows]] = False
                unavailable[active_rows] |= exclusions[active_rows, chosen[active_rows]]
                previous = chosen[:, None] if cache is not None else torch.cat((previous, chosen[:, None]), dim=1)

        steps = max(map(len, rollin)) + 1
        actions = torch.full((batch_size, steps), num_queries, dtype=torch.long, device=device)
        lengths = torch.tensor([len(order) for order in rollin], device=device)
        for image, order in enumerate(rollin):
            if order:
                actions[image, :len(order)] = torch.as_tensor(order, device=device)
        previous = torch.cat((
            torch.full((batch_size, 1), -1, dtype=torch.long, device=device),
            actions[:, :-1],
        ), dim=1)
        logits = self(mask_embeddings, image_features, regions, previous,
                      exclusions).float()
        remaining = target.clone()
        unavailable = torch.zeros_like(target)
        mask_loss_sum = logits.new_zeros(())
        eof_loss_sum = logits.new_zeros(())
        for step in range(steps):
            active = lengths > step
            at_eof = lengths == step
            allowed_logits = logits[:, step].masked_fill(
                torch.cat((unavailable, unavailable.new_zeros(batch_size, 1)), dim=1),
                -torch.inf,
            )
            log_probabilities = F.log_softmax(allowed_logits, dim=-1)
            # Inactive rows use EOF as a finite dummy target; their loss is ignored.
            target_actions = torch.cat((remaining, ~active[:, None]), dim=1)
            log_target_mass = torch.logsumexp(
                log_probabilities.masked_fill(~target_actions, -torch.inf), dim=-1
            )
            mask_loss_sum = mask_loss_sum - log_target_mass[active].sum()
            eof_loss_sum = eof_loss_sum - log_target_mass[at_eof].sum()
            chosen_rows = active.nonzero(as_tuple=True)[0]
            chosen = actions[chosen_rows, step]
            remaining[chosen_rows, chosen] = False
            unavailable[chosen_rows] |= exclusions[chosen_rows, chosen]
        mask_count = int(lengths.sum().item())
        token_count = mask_count + batch_size
        loss = (mask_loss_sum + eof_loss_sum) / token_count
        return loss, mask_loss_sum.detach(), eof_loss_sum.detach(), token_count, mask_count

    def rollout_permutation(self, mask_embeddings, image_features, mask_logits,
                            image_sizes, proposal_mask, sample=True, *, use_cache=True,
                            state_cache=None, image_ids=None):
        """Select every allowed proposal once; EOF is forced after the last mask."""
        batch_size, num_queries = mask_embeddings.shape[:2]
        if proposal_mask.shape != (batch_size, num_queries):
            raise ValueError("proposal_mask must have shape [batch, num_queries]")
        if proposal_mask.dtype != torch.bool:
            raise ValueError("proposal_mask must be boolean")
        regions = self.prepare_regions(mask_logits, image_sizes)
        exclusions = self.prepare_vocabulary_exclusions(mask_logits)
        selected = torch.zeros_like(proposal_mask)
        excluded_neighbors = torch.zeros_like(proposal_mask)
        counts = proposal_mask.sum(-1)
        previous = torch.full((batch_size, 1), -1, dtype=torch.long, device=mask_embeddings.device)
        action_steps = []
        log_probability = mask_embeddings.new_zeros(batch_size)
        max_steps = int(counts.max().item())
        cache = self._rollout_cache(use_cache, mask_embeddings, image_features, regions, exclusions, max_steps,
                                    state_cache=state_cache, image_ids=image_ids)
        for step in range(max_steps):
            active = counts > step
            logits = (self._decode_step(previous[:, -1], cache) if cache is not None else
                      self(mask_embeddings, image_features, regions, previous, exclusions,
                           state_cache=state_cache, image_ids=image_ids)[:, -1]).clone()
            remaining = proposal_mask & ~selected
            eligible = remaining & ~excluded_neighbors
            # IoU exclusions defer overlapping proposals. Once every remaining
            # proposal is deferred, restore them to finish the permutation.
            eligible = torch.where(eligible.any(-1, keepdim=True), eligible, remaining)
            logits[:, :num_queries].masked_fill_(~eligible, -torch.inf)
            logits[active, num_queries] = -torch.inf
            # Inactive rows still need a valid distribution for batched decoding.
            logits[~active, num_queries] = 0
            distribution = torch.distributions.Categorical(logits=logits)
            token = distribution.sample() if sample else logits.argmax(-1)
            if sample:
                log_probability = log_probability + torch.where(
                    active, distribution.log_prob(token), 0.0
                )
            action_steps.append(torch.where(active, token, -1).detach())
            chosen = active.nonzero(as_tuple=True)[0]
            selected[chosen, token[chosen]] = True
            excluded_neighbors[chosen] |= exclusions[chosen, token[chosen]]
            previous = token[:, None] if cache is not None else torch.cat((previous, token[:, None]), dim=1)
        rows = torch.stack(action_steps, 1).tolist() if action_steps else [[] for _ in range(batch_size)]
        orders = [[token for token in row if token >= 0] for row in rows]
        return orders, log_probability

    def permutation_log_probability(self, mask_embeddings, image_features, mask_logits,
                                    image_sizes, proposal_mask, orders, *, state_cache=None, image_ids=None):
        """Score fixed permutations from their prefix maps."""
        batch_size, num_queries = proposal_mask.shape
        counts = proposal_mask.sum(-1)
        steps = int(counts.max().item())
        if not steps:
            return self.eof.sum().expand(batch_size) * 0.0
        actions = torch.full(
            (batch_size, steps), num_queries, dtype=torch.long, device=proposal_mask.device
        )
        for index, order in enumerate(orders):
            if len(order) != int(counts[index]):
                raise ValueError("permutation length must equal the proposal count")
            actions[index, :len(order)] = torch.as_tensor(order, device=actions.device)
        previous = torch.cat((
            torch.full((batch_size, 1), -1, dtype=torch.long, device=actions.device),
            actions[:, :-1],
        ), dim=1)
        exclusions = self.prepare_vocabulary_exclusions(mask_logits)
        selected = torch.zeros_like(proposal_mask)
        excluded_neighbors = torch.zeros_like(proposal_mask)
        allowed = []
        for step in range(steps):
            active = counts > step
            remaining = proposal_mask & ~selected
            eligible = remaining & ~excluded_neighbors
            eligible = torch.where(eligible.any(-1, keepdim=True), eligible, remaining)
            allowed.append(torch.cat((eligible, (~active)[:, None]), dim=-1))
            chosen = active.nonzero(as_tuple=True)[0]
            selected[chosen, actions[chosen, step]] = True
            excluded_neighbors[chosen] |= exclusions[chosen, actions[chosen, step]]
        regions = self.prepare_regions(mask_logits, image_sizes)
        logits = self(mask_embeddings, image_features, regions, previous, exclusions,
                      state_cache=state_cache, image_ids=image_ids)
        logits = logits.float().masked_fill(~torch.stack(allowed, dim=1), -torch.inf)
        log_probabilities = F.log_softmax(logits, dim=-1).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        return torch.where(
            torch.arange(steps, device=actions.device)[None, :] < counts[:, None],
            log_probabilities, 0.0,
        ).sum(-1)
