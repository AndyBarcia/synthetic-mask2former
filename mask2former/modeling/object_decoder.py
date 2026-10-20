"""Map-conditioned pointer decoder over mask query embeddings."""

import torch
from torch import nn
from torch.nn import functional as F
from .map_encoder import MapEncoder
from .mask_groups import mask_compatibility, group_logits


class ObjectDecoder(nn.Module):
    def __init__(self, mask_dim, hidden_dim, num_queries, num_layers, num_heads,
                 dim_feedforward, mask_iou_threshold=0.8, prefix_tree_branches=4,
                 eof_threshold=.5,
                 collapse_masks=False, collapse_iou_threshold=.8):
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
        self.map_encoder = MapEncoder(hidden_dim, num_heads, dim_feedforward, num_queries)
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
                map_last_only=False):
        # Sampling and gradient scoring can share an outer AMP context.
        # Avoid reusing parameter casts cached by the preceding no-grad pass.
        with torch.autocast(device_type=mask_embeddings.device.type,
                enabled=torch.is_autocast_enabled(mask_embeddings.device.type), cache_enabled=False):
            return self._forward_impl(mask_embeddings, image_features, image_regions,
                previous_tokens, vocabulary_exclusions, positions=positions,
                ancestors=ancestors, return_hidden=return_hidden, map_last_only=map_last_only)

    def _forward_impl(self, mask_embeddings, image_features, image_regions, previous_tokens,
                      vocabulary_exclusions, *, positions=None, ancestors=None,
                      return_hidden=False, map_last_only=False):
        """Return next-token logits from independent map summaries."""
        if len(image_features) != 3 or len(image_regions) != 3:
            raise ValueError("Object decoder requires three image feature levels")
        mask_tokens = self.input_mask_projection(mask_embeddings)
        vocabulary = torch.cat((mask_tokens, self.eof.expand(mask_tokens.shape[0], 1, -1)), dim=1)
        previous = self.map_encoder(mask_tokens, image_features[0], image_regions[0],
            previous_tokens, ancestors, positions, last_only=map_last_only)
        steps = previous.shape[1]
        state_mask = ~torch.eye(steps, device=previous.device, dtype=torch.bool)
        vocabulary_mask = self._vocabulary_mask(vocabulary_exclusions, previous_tokens, ancestors)
        vocabulary_mask = vocabulary_mask[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            mask_tokens.shape[0] * self.num_heads, steps, self.num_queries + 1
        )
        decoded = previous
        for layer_index, layer in enumerate(self.layers.layers):
            level_index = layer_index % len(image_features)
            image_memory = torch.cat((
                image_features[level_index],
                self.null_image_token[level_index].expand(mask_tokens.shape[0], 1, -1),
            ), dim=1)
            decoded = layer(
                decoded, image_memory,
                tgt_mask=state_mask,
                memory_mask=self._memory_mask(image_regions[level_index], previous_tokens, ancestors),
            )
            attended = self.vocabulary_attention[layer_index](
                decoded, vocabulary, vocabulary, attn_mask=vocabulary_mask,
                need_weights=False,
            )[0]
            decoded = self.vocabulary_norm[layer_index](decoded + attended)
        decoded = self.norm(decoded)
        output_mask_tokens = self.output_mask_projection(mask_embeddings)
        output_vocabulary = torch.cat((
            output_mask_tokens, self.eof.expand(output_mask_tokens.shape[0], 1, -1),
        ), dim=1)
        logits = torch.matmul(decoded, output_vocabulary.transpose(1, 2)) * self.scale
        return (logits, decoded) if return_hidden else logits

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

    def _prepare_decode_cache(self, mask_embeddings, image_features, regions,
                              exclusions, max_steps):
        """Cache fixed projections and allocate incremental, rollout-local state.

        Projected tensors retain their gradients during differentiable rollouts.
        No cache survives a rollout or optimizer update.
        """
        if len(image_features) != 3 or len(regions) != 3:
            raise ValueError("Object decoder requires three image feature levels")
        batch_size = mask_embeddings.shape[0]
        vocabulary = torch.cat((self.input_mask_projection(mask_embeddings),
                                self.eof.expand(batch_size, 1, -1)), dim=1)
        output_vocabulary = torch.cat((self.output_mask_projection(mask_embeddings),
                                       self.eof.expand(batch_size, 1, -1)), dim=1)
        image_memory = [torch.cat((feature, self.null_image_token[level].expand(batch_size, 1, -1)), 1)
                        for level, feature in enumerate(image_features)]
        return {
            "vocabulary": vocabulary,
            "output_vocabulary": output_vocabulary,
            "image_kv": [self._project_attention_kv(layer.multihead_attn, image_memory[index % 3])
                         for index, layer in enumerate(self.layers.layers)],
            "vocabulary_kv": [self._project_attention_kv(attention, vocabulary)
                              for attention in self.vocabulary_attention],
            "features": image_features,
            "pool_cache": {},
            "history": [],
            "regions": regions,
            "exclusions": exclusions,
            "covered": [region.new_zeros(batch_size, region.shape[-1] + 1) for region in regions],
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
        batch_size, _, width = vocabulary.shape
        rows = torch.arange(batch_size, device=previous_token.device)
        valid = (previous_token >= 0) & (previous_token < self.num_queries)
        query_index = previous_token.clamp(0, self.num_queries - 1)
        cache["blocked"][:, :-1] |= cache["exclusions"][rows, query_index] & valid[:, None]
        for covered, regions in zip(cache["covered"], cache["regions"]):
            covered[:, :-1] |= regions[rows, query_index] & valid[:, None]
        cache["history"].append(previous_token)
        previous = torch.stack(cache["history"], 1)
        decoded = self.map_encoder(vocabulary[:, :-1], cache["features"][0], cache["regions"][0],
            previous, None, None, last_only=True, pool_cache=cache["pool_cache"])[:, -1:]
        for index, layer in enumerate(self.layers.layers):
            attended = layer.self_attn(decoded, decoded, decoded, need_weights=False)[0]
            decoded = layer.norm1(decoded + layer.dropout1(attended))
            attended = self._cached_cross_attention(layer.multihead_attn, decoded,
                                                     cache["image_kv"][index], cache["covered"][index % 3][:, None])
            decoded = layer.norm2(decoded + layer.dropout2(attended))
            decoded = layer.norm3(decoded + layer._ff_block(decoded))
            attended = self._cached_cross_attention(self.vocabulary_attention[index], decoded,
                                                     cache["vocabulary_kv"][index], cache["blocked"][:, None])
            decoded = self.vocabulary_norm[index](decoded + attended)
        cache["step"] += 1
        return (self.norm(decoded) @ cache["output_vocabulary"].transpose(1, 2))[:, 0] * self.scale

    def _rollout_cache(self, use_cache, mask_embeddings, image_features, regions,
                       exclusions, max_steps):
        # Respect custom forward implementations.
        if (use_cache
                and getattr(self.forward, "__func__", None) is ObjectDecoder.forward):
            return self._prepare_decode_cache(mask_embeddings, image_features, regions,
                                              exclusions, max_steps)
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
        order = torch.full((batch_size, num_queries), -1, dtype=torch.long, device=mask_embeddings.device)
        log_odds = torch.zeros(batch_size, num_queries, dtype=mask_embeddings.dtype, device=mask_embeddings.device)
        probabilities = torch.zeros(batch_size, num_queries, dtype=mask_embeddings.dtype, device=mask_embeddings.device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=mask_embeddings.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=mask_embeddings.device)
        tokens = torch.full((batch_size, 1), -1, dtype=torch.long, device=mask_embeddings.device)
        cache = self._rollout_cache(use_cache, mask_embeddings, image_features, image_regions, exclusions, num_queries)
        for _ in range(num_queries):
            logits = (self._decode_step(tokens[:, -1], cache) if cache is not None else
                      self(mask_embeddings, image_features, image_regions, tokens, exclusions,
                           map_last_only=True)[:, -1])
            logits[:, :num_queries].masked_fill_(unavailable, -torch.inf)
            selection_logits = group_logits(logits.float(), compatibility) if compatibility is not None else logits
            next_token = self._greedy_action(selection_logits)
            active = ~finished & (next_token != num_queries)
            active_batch = torch.arange(batch_size, device=unavailable.device)[active]
            unavailable[active_batch] |= exclusions[active_batch, next_token[active]]
            order[active_batch, lengths[active]] = next_token[active]
            relative_logits = selection_logits[active_batch, next_token[active]] - selection_logits[active_batch, num_queries]
            log_odds[active_batch, lengths[active]] = relative_logits.to(log_odds.dtype)
            probabilities[active_batch, lengths[active]] = relative_logits.sigmoid().to(probabilities.dtype)
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

    def rollout(self, mask_embeddings, image_features, mask_logits, image_sizes,
                sample=True, max_steps=32, return_diagnostics=False,
                return_actions=False, query_bias_logits=None, *, use_cache=True, greedy_eof_threshold=None,
                query_classes=None):
        """Generate query orders and sequence log probabilities for policy gradients.

        A capped rollout is treated as an implicit EOF after the last selected query.
        """
        batch_size, num_queries = mask_embeddings.shape[:2]
        max_steps = min(max_steps, num_queries + 1)
        regions = self.prepare_regions(mask_logits, image_sizes)
        exclusions = self.prepare_vocabulary_exclusions(mask_logits)
        compatibility = (mask_compatibility(mask_logits, query_classes, self.collapse_iou_threshold)
                         if self.collapse_masks else None)
        unavailable = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=mask_embeddings.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=mask_embeddings.device)
        previous = torch.full((batch_size, 1), -1, dtype=torch.long, device=mask_embeddings.device)
        action_steps = []
        log_probability = mask_embeddings.new_zeros(batch_size)
        diagnostics = {
            key: torch.zeros(batch_size, device=mask_embeddings.device)
            for key in ("actions", "entropy", "top1_probability", "eof_probability", "available_masks")
        } if return_diagnostics else None
        cache = self._rollout_cache(use_cache, mask_embeddings, image_features, regions, exclusions, max_steps)
        for _ in range(max_steps):
            logits = (self._decode_step(previous[:, -1], cache) if cache is not None else
                      self(mask_embeddings, image_features, regions, previous, exclusions,
                           map_last_only=True)[:, -1]).clone()
            logits[:, :num_queries].masked_fill_(unavailable.clone(), -torch.inf)
            if compatibility is not None:
                logits = group_logits(logits.float(), compatibility)
            distribution = torch.distributions.Categorical(logits=logits)
            token = distribution.sample() if sample else self._greedy_action(logits, greedy_eof_threshold)
            active = ~finished
            if diagnostics is not None:
                with torch.no_grad():
                    probabilities = distribution.probs.detach()
                    diagnostics["actions"] += active.float()
                    diagnostics["entropy"] += torch.where(active, distribution.entropy().detach(), 0)
                    diagnostics["top1_probability"] += torch.where(active, probabilities.max(-1).values, 0)
                    diagnostics["eof_probability"] += torch.where(active, probabilities[:, -1], 0)
                    diagnostics["available_masks"] += torch.where(active, (~unavailable).sum(-1), 0)
            if sample:
                log_probability = log_probability + torch.where(
                    active, distribution.log_prob(token), 0.0
                )
            action_steps.append(torch.where(active, token, -1).detach())
            chosen = active & (token != num_queries)
            chosen_batch = torch.arange(batch_size, device=unavailable.device)[chosen]
            unavailable[chosen_batch] |= exclusions[chosen_batch, token[chosen]]
            finished = finished | (token == num_queries)
            if finished.all():
                break
            next_input = torch.where(finished, num_queries, token)[:, None]
            previous = next_input if cache is not None else torch.cat((previous, next_input), dim=1)
        # Transfer the complete ragged trajectories once, rather than each token.
        rows = torch.stack(action_steps, 1).tolist() if action_steps else [[] for _ in range(batch_size)]
        actions = [[token for token in row if token >= 0] for row in rows]
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
                                       query_bias_logits=None, *, query_classes=None):
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
        logits = self(mask_embeddings, image_features, regions, previous, exclusions)
        logits = logits.float().masked_fill(~torch.stack(allowed, dim=1), -torch.inf)
        if self.collapse_masks:
            logits = group_logits(logits, mask_compatibility(
                mask_logits, query_classes, self.collapse_iou_threshold))
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
                            image_sizes, proposal_mask, sample=True, *, use_cache=True):
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
        cache = self._rollout_cache(use_cache, mask_embeddings, image_features, regions, exclusions, max_steps)
        for step in range(max_steps):
            active = counts > step
            logits = (self._decode_step(previous[:, -1], cache) if cache is not None else
                      self(mask_embeddings, image_features, regions, previous, exclusions)[:, -1]).clone()
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
                                    image_sizes, proposal_mask, orders):
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
        logits = self(mask_embeddings, image_features, regions, previous, exclusions)
        logits = logits.float().masked_fill(~torch.stack(allowed, dim=1), -torch.inf)
        log_probabilities = F.log_softmax(logits, dim=-1).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        return torch.where(
            torch.arange(steps, device=actions.device)[None, :] < counts[:, None],
            log_probabilities, 0.0,
        ).sum(-1)
