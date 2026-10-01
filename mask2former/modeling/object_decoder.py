"""Autoregressive pointer decoder over the mask query embeddings."""

import torch
from torch import nn
from torch.nn import functional as F


class ObjectDecoder(nn.Module):
    def __init__(self, mask_dim, hidden_dim, num_queries, num_layers, num_heads,
                 dim_feedforward, mask_iou_threshold=0.8, prefix_tree_branches=4):
        super().__init__()
        if not 0 <= mask_iou_threshold <= 1:
            raise ValueError("mask_iou_threshold must be between 0 and 1")
        if prefix_tree_branches < 1:
            raise ValueError("prefix_tree_branches must be positive")
        self.prefix_tree_branches = prefix_tree_branches
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
    def pack_prefix_trees(sequences, num_queries, device):
        """Pack paths into context tries; edge counts preserve expanded-path CE.

        Each node represents a prefix (root is BOS). Targets are outgoing edge
        labels, including EOF. Shared nodes accumulate one count per path.
        Padding attends only to itself and has no loss weight.
        """
        trees = []
        for paths in sequences:
            tokens, parents, depths, counts = [-1], [-1], [0], [{}]
            children = {}
            for path in paths:
                node = 0
                for token in list(path) + [num_queries]:
                    token = int(token)
                    counts[node][token] = counts[node].get(token, 0) + 1
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
        """Block the union of masks emitted up to each autoregressive position."""
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
                vocabulary_exclusions, *, positions=None, ancestors=None):
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
        previous = previous + (self.position.weight[:steps] if positions is None else self.position(positions))
        causal_mask = torch.ones(steps, steps, dtype=torch.bool, device=previous.device).triu(1)
        if ancestors is not None:
            causal_mask = (~ancestors)[:, None].expand(-1, self.num_heads, -1, -1).reshape(
                mask_tokens.shape[0] * self.num_heads, steps, steps
            )
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
                tgt_mask=causal_mask,
                memory_mask=self._memory_mask(image_regions[level_index], previous_tokens, ancestors),
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
