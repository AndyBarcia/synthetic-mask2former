"""Object-decoder policy over dynamic mask proposals, without overlap exclusions."""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

from ..object_decoder import ObjectDecoder


class ProposalObjectDecoder(ObjectDecoder):
    """Exclude only previously emitted IDs; leave overlap decisions to the policy."""

    def prepare_vocabulary_exclusions(self, mask_logits):
        batch, queries = mask_logits.shape[:2]
        return torch.eye(queries, device=mask_logits.device, dtype=torch.bool)[None].expand(batch, -1, -1)


@torch.no_grad()
def match_masks(proposal, target, threshold):
    valid = target['masks'].flatten(1).any(1)
    gt = F.interpolate(target['masks'][valid, None].float(), proposal['masks'].shape[-2:], mode='nearest')[:, 0]
    if not len(gt):
        return torch.empty(0, device=gt.device, dtype=torch.long)
    predicted = (proposal['masks'] > 0).flatten(1).float()
    gt = gt.flatten(1)
    intersection = predicted @ gt.T
    iou = intersection / (predicted.sum(-1)[:, None] + gt.sum(-1)[None] - intersection).clamp_min(1)
    correct_class = proposal['labels'][:, None] == target['labels'][valid][None]
    rows, cols = linear_sum_assignment((-iou + 1000 * ~correct_class).cpu().numpy())
    rows = torch.as_tensor(rows, device=gt.device)
    cols = torch.as_tensor(cols, device=gt.device)
    return rows[correct_class[rows, cols] & (iou[rows, cols] > threshold)]


class DynamicMaskPolicy(nn.Module):
    def __init__(self, decoder, num_classes, mask_dim, match_threshold=0.5):
        super().__init__()
        self.decoder = decoder
        self.num_classes = num_classes
        self.match_threshold = match_threshold
        # 169 dynamic parameters, center/stride coordinates, bias, score, class.
        self.adapter = nn.Sequential(nn.LayerNorm(175 + num_classes), nn.Linear(175 + num_classes, mask_dim))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def tokens(self, proposal, size):
        scale = proposal['centers'].new_tensor((size[1], size[0]))
        context = torch.cat((proposal['dynamic_parameters'], proposal['centers'] / scale,
                             proposal['strides'] / scale, proposal['bias'], proposal['scores'][:, None],
                             F.one_hot(proposal['labels'], self.num_classes).float()), -1)
        return proposal['embeddings'] + self.adapter(context)

    def supervised_loss(self, proposals, targets, memories, sizes, padded_size):
        masks = torch.stack([p['masks'] for p in proposals])
        embeddings = torch.stack([self.tokens(p, padded_size) for p in proposals])
        paths = []
        for proposal, target in zip(proposals, targets):
            matched = match_masks(proposal, target, self.match_threshold)
            paths.append([matched[torch.randperm(len(matched), device=masks.device)].tolist()
                          for _ in range(self.decoder.prefix_tree_branches)])
        previous, positions, ancestors, weights = self.decoder.pack_prefix_trees(paths, self.decoder.num_queries, masks.device)
        regions = self.decoder.prepare_regions(masks, sizes)
        logits = self.decoder(embeddings, memories, regions, previous,
                              self.decoder.prepare_vocabulary_exclusions(masks), positions=positions, ancestors=ancestors)
        return -(logits.float().log_softmax(-1) * weights).sum() / weights.sum()

    @torch.no_grad()
    def select(self, proposals, memories, sizes, padded_size, max_steps):
        masks = torch.stack([p['masks'] for p in proposals])
        embeddings = torch.stack([self.tokens(p, padded_size) for p in proposals])
        orders, _ = self.decoder.rollout(embeddings, memories, masks, sizes, sample=False, max_steps=max_steps)
        return orders
