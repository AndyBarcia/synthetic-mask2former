"""Autoregressive pointer decoder over the mask query embeddings."""

import torch
from torch import nn
from torch.nn import functional as F


class ObjectDecoder(nn.Module):
    def __init__(self, mask_dim, hidden_dim, num_queries, num_layers, num_heads, dim_feedforward):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
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

    def forward(self, mask_embeddings, image_features, image_regions, previous_tokens):
        """Return next-token logits with causal multi-scale image cross-attention."""
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
        decoded = self.norm(decoded)
        return torch.matmul(decoded, vocabulary.transpose(1, 2)) * self.scale

    @torch.no_grad()
    def generate(self, mask_embeddings, image_features, mask_logits, image_sizes):
        """Return ordered queries, mask-versus-EOF logits, and probabilities."""
        batch_size, num_queries, _ = mask_embeddings.shape
        image_regions = self.prepare_regions(mask_logits, image_sizes)
        selected = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=mask_embeddings.device)
        order = torch.full((batch_size, num_queries), -1, dtype=torch.long, device=mask_embeddings.device)
        log_odds = torch.zeros(batch_size, num_queries, dtype=mask_embeddings.dtype, device=mask_embeddings.device)
        probabilities = torch.zeros(batch_size, num_queries, dtype=mask_embeddings.dtype, device=mask_embeddings.device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=mask_embeddings.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=mask_embeddings.device)
        tokens = torch.full((batch_size, 1), -1, dtype=torch.long, device=mask_embeddings.device)
        for _ in range(num_queries):
            logits = self(mask_embeddings, image_features, image_regions, tokens)[:, -1]
            logits[:, :num_queries].masked_fill_(selected, -torch.inf)
            next_token = logits.argmax(-1)
            active = ~finished & (next_token != num_queries)
            active_batch = torch.arange(batch_size, device=selected.device)[active]
            selected[active_batch, next_token[active]] = True
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
