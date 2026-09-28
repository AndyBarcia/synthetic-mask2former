"""Autoregressive pointer decoder over the mask query embeddings."""

import torch
from torch import nn
from torch.nn import functional as F


class ObjectDecoder(nn.Module):
    def __init__(self, mask_dim, hidden_dim, num_queries, num_layers, num_heads,
                 dim_feedforward, mask_iou_threshold=0.8,
                 bias_order_constraint=False):
        super().__init__()
        if not 0 <= mask_iou_threshold <= 1:
            raise ValueError("mask_iou_threshold must be between 0 and 1")
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.mask_iou_threshold = mask_iou_threshold
        self.bias_order_constraint = bias_order_constraint
        self.mask_projection = nn.Linear(mask_dim, hidden_dim)
        self.bos = nn.Parameter(torch.zeros(hidden_dim))
        self.eof = nn.Parameter(torch.zeros(hidden_dim))
        # This key remains available if previous masks cover an entire feature map.
        self.null_image_token = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.normal_(self.null_image_token, std=0.02)
        self.position = nn.Embedding(num_queries + 1, hidden_dim)
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
        # The query prior is shared by vocabulary attention and token selection.
        # Its input is detached so object decoding cannot train the mask decoder.
        self.query_bias_scale = nn.Parameter(torch.ones(()))
        self.query_bias_offset = nn.Parameter(torch.zeros(()))

    @staticmethod
    def prepare_regions(mask_logits, image_sizes):
        """Threshold final query masks at each cross-attention feature scale."""
        with torch.no_grad():
            return [
                (F.interpolate(mask_logits.float(), size=size, mode="bilinear", align_corners=False)
                 .flatten(2) > 0)
                for size in image_sizes
            ]

    def prepare_vocabulary_exclusions(self, mask_logits, query_bias_logits=None):
        """Exclude overlaps and higher-priority queries after each selection."""
        with torch.no_grad():
            masks = (mask_logits.detach().flatten(2) > 0).float()
            intersection = torch.bmm(masks, masks.transpose(1, 2))
            area = masks.sum(-1)
            union = area[:, :, None] + area[:, None, :] - intersection
            exclusions = intersection / union.clamp_min(1) > self.mask_iou_threshold
            identity = torch.eye(masks.shape[1], dtype=torch.bool, device=masks.device)
            exclusions = exclusions | identity
            if self.bias_order_constraint:
                if query_bias_logits is None:
                    raise ValueError("Bias-ordered decoding requires query_bias_logits")
                bias = query_bias_logits.detach()
                # Row i lists queries unavailable after selecting query i.
                exclusions = exclusions | (bias[:, None, :] > bias[:, :, None])
            return exclusions

    def _query_prior(self, query_bias_logits, vocabulary):
        if query_bias_logits is None:
            return vocabulary.new_zeros(vocabulary.shape[0], self.num_queries + 1)
        prior = (query_bias_logits.detach().to(vocabulary.dtype) * self.query_bias_scale
                 + self.query_bias_offset).to(vocabulary.dtype)
        return F.pad(prior, (0, 1), value=0.0)

    def _bias_ranks(self, query_bias_logits, batch_size, device):
        if not self.bias_order_constraint:
            return None
        if query_bias_logits is None or query_bias_logits.shape != (batch_size, self.num_queries):
            raise ValueError("Bias-ordered decoding requires query_bias_logits [batch, queries]")
        order = query_bias_logits.detach().argsort(dim=-1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(self.num_queries, device=device).expand_as(order))
        return ranks

    @staticmethod
    def proposals_from_query_bias(query_bias_logits, threshold):
        """Select predicted proposals without GT counts; keep one if none pass."""
        probabilities = query_bias_logits.sigmoid()
        proposals = probabilities >= threshold
        empty = ~proposals.any(-1)
        if empty.any():
            proposals[empty, probabilities[empty].argmax(-1)] = True
        return proposals

    @staticmethod
    def _vocabulary_mask(exclusions, previous_tokens):
        """Exclude selected masks and their IoU neighbors from later positions."""
        batch_size, num_queries, _ = exclusions.shape
        indices = previous_tokens.clamp(0, num_queries - 1)
        prior = exclusions.gather(1, indices.unsqueeze(-1).expand(-1, -1, num_queries))
        valid = (previous_tokens >= 0) & (previous_tokens < num_queries)
        blocked = (prior & valid.unsqueeze(-1)).cumsum(dim=1) > 0
        # EOF remains available even when every mask has been excluded.
        return F.pad(blocked, (0, 1), value=False)

    def _memory_mask(self, regions, previous_tokens):
        """Block the union of masks emitted up to each autoregressive position."""
        batch_size, num_queries, spatial_tokens = regions.shape
        indices = previous_tokens.clamp(0, num_queries - 1)
        prior_regions = regions.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, spatial_tokens)
        )
        valid = (previous_tokens >= 0) & (previous_tokens < num_queries)
        covered = (prior_regions & valid.unsqueeze(-1)).cumsum(dim=1) > 0
        # The learned null image key must never be masked, including at full coverage.
        covered = F.pad(covered, (0, 1), value=False)
        return covered[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            batch_size * self.num_heads, previous_tokens.shape[1], spatial_tokens + 1
        )

    def forward(self, mask_embeddings, image_features, image_regions, previous_tokens,
                vocabulary_exclusions, query_bias_logits=None, mask_token_logits=True):
        """Return next-token logits with causal image and vocabulary attention."""
        if len(image_features) != 3 or len(image_regions) != 3:
            raise ValueError("Object decoder requires three image feature levels")
        mask_tokens = self.mask_projection(mask_embeddings)
        vocabulary = torch.cat((mask_tokens, self.eof.expand(mask_tokens.shape[0], 1, -1)), dim=1)
        previous = vocabulary.gather(
            1, previous_tokens.clamp_min(0).unsqueeze(-1).expand(-1, -1, vocabulary.shape[-1])
        )
        previous = torch.where(
            (previous_tokens < 0).unsqueeze(-1), self.bos, previous
        )
        steps = previous.shape[1]
        previous = previous + self.position.weight[:steps]
        causal_mask = torch.ones(steps, steps, dtype=torch.bool, device=previous.device).triu(1)
        blocked = self._vocabulary_mask(vocabulary_exclusions, previous_tokens)
        query_prior = self._query_prior(query_bias_logits, vocabulary)
        vocabulary_mask = query_prior[:, None, :].expand(-1, steps, -1).masked_fill(
            blocked, -torch.inf
        )
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
                tgt_mask=causal_mask,
                memory_mask=self._memory_mask(image_regions[level_index], previous_tokens),
            )
            attended = self.vocabulary_attention[layer_index](
                decoded, vocabulary, vocabulary, attn_mask=vocabulary_mask,
                need_weights=False,
            )[0]
            decoded = self.vocabulary_norm[layer_index](decoded + attended)
        decoded = self.norm(decoded)
        logits = torch.matmul(decoded, vocabulary.transpose(1, 2)) * self.scale
        logits = logits + query_prior[:, None, :]
        return logits.masked_fill(blocked, -torch.inf) if mask_token_logits else logits

    @torch.no_grad()
    def generate(self, mask_embeddings, image_features, mask_logits, image_sizes,
                 query_bias_logits=None):
        """Return ordered queries, mask-versus-EOF logits, and probabilities."""
        batch_size, num_queries, _ = mask_embeddings.shape
        image_regions = self.prepare_regions(mask_logits, image_sizes)
        exclusions = self.prepare_vocabulary_exclusions(mask_logits, query_bias_logits)
        bias_ranks = self._bias_ranks(query_bias_logits, batch_size, mask_embeddings.device)
        unavailable = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=mask_embeddings.device)
        order = torch.full((batch_size, num_queries), -1, dtype=torch.long, device=mask_embeddings.device)
        log_odds = torch.zeros(batch_size, num_queries, dtype=mask_embeddings.dtype, device=mask_embeddings.device)
        probabilities = torch.zeros(batch_size, num_queries, dtype=mask_embeddings.dtype, device=mask_embeddings.device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=mask_embeddings.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=mask_embeddings.device)
        tokens = torch.full((batch_size, 1), -1, dtype=torch.long, device=mask_embeddings.device)
        for _ in range(num_queries):
            logits = self(mask_embeddings, image_features, image_regions, tokens, exclusions,
                          query_bias_logits)[:, -1]
            logits[:, :num_queries].masked_fill_(unavailable, -torch.inf)
            next_token = logits.argmax(-1)
            active = ~finished & (next_token != num_queries)
            active_batch = torch.arange(batch_size, device=unavailable.device)[active]
            unavailable[active_batch] |= exclusions[active_batch, next_token[active]]
            if bias_ranks is not None:
                chosen_ranks = bias_ranks[active_batch, next_token[active]]
                unavailable[active_batch] |= bias_ranks[active_batch] <= chosen_ranks[:, None]
            order[active_batch, lengths[active]] = next_token[active]
            relative_logits = logits[active_batch, next_token[active]] - logits[active_batch, num_queries]
            log_odds[active_batch, lengths[active]] = relative_logits
            probabilities[active_batch, lengths[active]] = relative_logits.sigmoid()
            lengths += active.long()
            finished |= next_token == num_queries
            if finished.all():
                break
            tokens = torch.cat((tokens, torch.where(finished, num_queries, next_token)[:, None]), dim=1)
        return [
            (order[batch_index, :lengths[batch_index]],
             log_odds[batch_index, :lengths[batch_index]],
             probabilities[batch_index, :lengths[batch_index]])
            for batch_index in range(batch_size)
        ]

    def rollout(self, mask_embeddings, image_features, mask_logits, image_sizes,
                sample=True, max_steps=32, return_diagnostics=False,
                return_actions=False, query_bias_logits=None):
        """Generate query orders and sequence log probabilities for policy gradients.

        A capped rollout is treated as an implicit EOF after the last selected query.
        """
        batch_size, num_queries = mask_embeddings.shape[:2]
        max_steps = min(max_steps, num_queries + 1)
        regions = self.prepare_regions(mask_logits, image_sizes)
        exclusions = self.prepare_vocabulary_exclusions(mask_logits, query_bias_logits)
        bias_ranks = self._bias_ranks(query_bias_logits, batch_size, mask_embeddings.device)
        unavailable = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=mask_embeddings.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=mask_embeddings.device)
        previous = torch.full((batch_size, 1), -1, dtype=torch.long, device=mask_embeddings.device)
        orders = [[] for _ in range(batch_size)]
        actions = [[] for _ in range(batch_size)] if return_actions else None
        log_probability = mask_embeddings.new_zeros(batch_size)
        diagnostics = {
            key: torch.zeros(batch_size, device=mask_embeddings.device)
            for key in ("actions", "entropy", "top1_probability", "eof_probability", "available_masks")
        } if return_diagnostics else None
        for _ in range(max_steps):
            logits = self(mask_embeddings, image_features, regions, previous, exclusions,
                          query_bias_logits)[:, -1].clone()
            logits[:, :num_queries].masked_fill_(unavailable.clone(), -torch.inf)
            distribution = torch.distributions.Categorical(logits=logits)
            token = distribution.sample() if sample else logits.argmax(-1)
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
            for index in range(batch_size):
                if active[index]:
                    if actions is not None:
                        actions[index].append(int(token[index]))
                    if token[index] != num_queries:
                        orders[index].append(int(token[index]))
            chosen = active & (token != num_queries)
            chosen_batch = torch.arange(batch_size, device=unavailable.device)[chosen]
            unavailable[chosen_batch] |= exclusions[chosen_batch, token[chosen]]
            if bias_ranks is not None:
                chosen_ranks = bias_ranks[chosen_batch, token[chosen]]
                unavailable[chosen_batch] |= bias_ranks[chosen_batch] <= chosen_ranks[:, None]
            finished = finished | (token == num_queries)
            if finished.all():
                break
            previous = torch.cat((previous, torch.where(finished, num_queries, token)[:, None]), dim=1)
        if diagnostics is not None:
            diagnostics["length"] = torch.tensor([len(order) for order in orders], device=finished.device).float()
            diagnostics["truncated"] = (~finished).float()
            if return_actions:
                return orders, log_probability, diagnostics, actions
            return orders, log_probability, diagnostics
        if return_actions:
            return orders, log_probability, actions
        return orders, log_probability

    def autoregressive_log_probability(self, mask_embeddings, image_features,
                                       mask_logits, image_sizes, trajectories,
                                       query_bias_logits=None):
        """Score sampled mask and EOF actions in one causal decoder pass."""
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
        exclusions = self.prepare_vocabulary_exclusions(mask_logits, query_bias_logits)
        bias_ranks = self._bias_ranks(query_bias_logits, batch_size, mask_embeddings.device)
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
            if bias_ranks is not None:
                chosen_ranks = bias_ranks[chosen_batch, actions[chosen_batch, step]]
                unavailable[chosen_batch] |= bias_ranks[chosen_batch] <= chosen_ranks[:, None]
        regions = self.prepare_regions(mask_logits, image_sizes)
        logits = self(mask_embeddings, image_features, regions, previous, exclusions,
                      query_bias_logits)
        logits = logits.float().masked_fill(~torch.stack(allowed, dim=1), -torch.inf)
        log_probabilities = F.log_softmax(logits, dim=-1).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        return torch.where(
            torch.arange(steps, device=actions.device)[None, :] < lengths[:, None],
            log_probabilities, 0.0,
        ).sum(-1)

    def set_imitation_loss(self, mask_embeddings, image_features, mask_logits,
                           image_sizes, target_sets):
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
            for _ in range(int(target.sum(-1).max().item())):
                active = remaining.any(-1)
                eligible = remaining & ~unavailable
                if (active & ~eligible.any(-1)).any():
                    raise ValueError("Target set contains mutually excluded queries")
                logits = self(mask_embeddings, image_features, regions, previous,
                              exclusions)[:, -1].float()
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
                previous = torch.cat((previous, chosen[:, None]), dim=1)

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
                            image_sizes, proposal_mask, sample=True):
        """Select every allowed proposal once; EOF is forced after the last mask."""
        if self.bias_order_constraint:
            raise ValueError("Bias-ordered decoding supports free EOF rollouts only")
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
        orders = [[] for _ in range(batch_size)]
        log_probability = mask_embeddings.new_zeros(batch_size)
        for step in range(int(counts.max().item())):
            active = counts > step
            logits = self(mask_embeddings, image_features, regions, previous, exclusions,
                          mask_token_logits=False)[:, -1].clone()
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
            for index in range(batch_size):
                if active[index]:
                    orders[index].append(int(token[index]))
            chosen = active.nonzero(as_tuple=True)[0]
            selected[chosen, token[chosen]] = True
            excluded_neighbors[chosen] |= exclusions[chosen, token[chosen]]
            previous = torch.cat((previous, token[:, None]), dim=1)
        return orders, log_probability

    def permutation_log_probability(self, mask_embeddings, image_features, mask_logits,
                                    image_sizes, proposal_mask, orders):
        """Score fixed permutations in one causal decoder pass per batch."""
        if self.bias_order_constraint:
            raise ValueError("Bias-ordered decoding supports free EOF rollouts only")
        batch_size, num_queries = proposal_mask.shape
        counts = proposal_mask.sum(-1)
        steps = int(counts.max().item())
        if not steps:
            return self.bos.sum().expand(batch_size) * 0.0
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
                      mask_token_logits=False)
        logits = logits.float().masked_fill(~torch.stack(allowed, dim=1), -torch.inf)
        log_probabilities = F.log_softmax(logits, dim=-1).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        return torch.where(
            torch.arange(steps, device=actions.device)[None, :] < counts[:, None],
            log_probabilities, 0.0,
        ).sum(-1)
