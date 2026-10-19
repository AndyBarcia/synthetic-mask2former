"""Probability-preserving, same-class complete-link mask groups."""
import numpy as np
import torch


@torch.no_grad()
def mask_compatibility(mask_logits, query_classes, threshold):
    if query_classes is None or query_classes.shape != mask_logits.shape[:2]:
        raise ValueError("Mask collapsing requires query_classes with shape [batch, queries]")
    masks = (mask_logits.detach().flatten(2) > 0).float()
    intersection = masks @ masks.transpose(1, 2)
    area = masks.sum(-1)
    iou = intersection / (area[:, :, None] + area[:, None, :] - intersection).clamp_min(1)
    same_class = query_classes[:, :, None] == query_classes[:, None, :]
    return ((iou > threshold) & same_class).cpu().numpy()


def group_logits(logits, compatibility):
    """Log-sum-exp each group at its highest-logit representative's index.

    Groups are built greedily in descending individual-logit order, with query
    index breaking ties. Every pair in a group must be compatible (no chains).
    EOF is its own action. Group topology is detached; probability sums remain
    differentiable. Supports [B,Q+1] and [B,T,Q+1] logits.
    """
    batch, queries = compatibility.shape[:2]
    if logits.shape[0] != batch or logits.shape[-1] != queries + 1:
        raise ValueError("Grouped logits and mask compatibility shapes differ")
    per_image = logits.reshape(batch, -1, queries + 1)
    result = []
    for image, states in enumerate(per_image):
        for row in states:
            raw = row.detach().float().cpu().numpy()
            remaining = sorted(np.flatnonzero(np.isfinite(raw[:-1])), key=lambda q: (-raw[q], q))
            representatives, masses = [], []
            while remaining:
                anchor = remaining.pop(0)
                group, rest = [anchor], []
                for query in remaining:
                    (group if compatibility[image, query, group].all() else rest).append(query)
                remaining = rest
                representatives.append(int(anchor))
                masses.append(torch.logsumexp(row[group], dim=0))
            representatives.append(queries)
            masses.append(row[-1])
            indices = torch.tensor(representatives, device=row.device)
            result.append(torch.full_like(row, -torch.inf).scatter(0, indices, torch.stack(masses)))
    return torch.stack(result).reshape_as(logits)
