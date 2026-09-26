"""Autoregressive pointer decoder over the mask query embeddings."""

import torch
from torch import nn
from torch.nn import functional as F


class ObjectDecoder(nn.Module):
    def __init__(self, mask_dim, hidden_dim, num_queries, num_layers, num_heads,
                 dim_feedforward, mask_iou_threshold=0.8):
        super().__init__()
        if not 0 <= mask_iou_threshold <= 1:
            raise ValueError("mask_iou_threshold must be between 0 and 1")
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.mask_iou_threshold = mask_iou_threshold
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
                vocabulary_exclusions):
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
        vocabulary_mask = self._vocabulary_mask(vocabulary_exclusions, previous_tokens)
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
        return torch.matmul(decoded, vocabulary.transpose(1, 2)) * self.scale

    @torch.no_grad()
    def generate(self, mask_embeddings, image_features, mask_logits, image_sizes):
        """Return ordered queries, mask-versus-EOF logits, and probabilities."""
        batch_size, num_queries, _ = mask_embeddings.shape
        image_regions = self.prepare_regions(mask_logits, image_sizes)
        exclusions = self.prepare_vocabulary_exclusions(mask_logits)
        unavailable = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=mask_embeddings.device)
        order = torch.full((batch_size, num_queries), -1, dtype=torch.long, device=mask_embeddings.device)
        log_odds = torch.zeros(batch_size, num_queries, dtype=mask_embeddings.dtype, device=mask_embeddings.device)
        probabilities = torch.zeros(batch_size, num_queries, dtype=mask_embeddings.dtype, device=mask_embeddings.device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=mask_embeddings.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=mask_embeddings.device)
        tokens = torch.full((batch_size, 1), -1, dtype=torch.long, device=mask_embeddings.device)
        for _ in range(num_queries):
            logits = self(mask_embeddings, image_features, image_regions, tokens, exclusions)[:, -1]
            logits[:, :num_queries].masked_fill_(unavailable, -torch.inf)
            next_token = logits.argmax(-1)
            active = ~finished & (next_token != num_queries)
            active_batch = torch.arange(batch_size, device=unavailable.device)[active]
            unavailable[active_batch] |= exclusions[active_batch, next_token[active]]
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
                sample=True, max_steps=32, return_diagnostics=False):
        """Generate query orders and sequence log probabilities for policy gradients.

        A capped rollout is treated as an implicit EOF after the last selected query.
        """
        batch_size, num_queries = mask_embeddings.shape[:2]
        max_steps = min(max_steps, num_queries + 1)
        regions = self.prepare_regions(mask_logits, image_sizes)
        exclusions = self.prepare_vocabulary_exclusions(mask_logits)
        unavailable = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=mask_embeddings.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=mask_embeddings.device)
        previous = torch.full((batch_size, 1), -1, dtype=torch.long, device=mask_embeddings.device)
        orders = [[] for _ in range(batch_size)]
        log_probability = mask_embeddings.new_zeros(batch_size)
        diagnostics = {
            key: torch.zeros(batch_size, device=mask_embeddings.device)
            for key in ("actions", "entropy", "top1_probability", "eof_probability", "available_masks")
        } if return_diagnostics else None
        for _ in range(max_steps):
            logits = self(mask_embeddings, image_features, regions, previous, exclusions)[:, -1].clone()
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
                if active[index] and token[index] != num_queries:
                    orders[index].append(int(token[index]))
            chosen = active & (token != num_queries)
            chosen_batch = torch.arange(batch_size, device=unavailable.device)[chosen]
            unavailable[chosen_batch] |= exclusions[chosen_batch, token[chosen]]
            finished = finished | (token == num_queries)
            if finished.all():
                break
            previous = torch.cat((previous, torch.where(finished, num_queries, token)[:, None]), dim=1)
        if diagnostics is not None:
            diagnostics["length"] = torch.tensor([len(order) for order in orders], device=finished.device).float()
            diagnostics["truncated"] = (~finished).float()
            return orders, log_probability, diagnostics
        return orders, log_probability
