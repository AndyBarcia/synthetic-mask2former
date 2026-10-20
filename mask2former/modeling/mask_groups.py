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
    # Most images have no collapsible pair. Their logits already are grouped.
    off_diagonal = compatibility.copy()
    diagonal = np.arange(queries)
    off_diagonal[:, diagonal, diagonal] = False
    if not off_diagonal.any():
        return logits
    per_image = logits.reshape(batch, -1, queries + 1)
    # Transfer topology once, rather than once per state and once per group.
    raw = per_image.detach().float().cpu().numpy()
    labels = np.broadcast_to(np.arange(queries + 1), raw.shape).copy()
    for image in range(batch):
        for state, row in enumerate(raw[image]):
            remaining = sorted(np.flatnonzero(np.isfinite(row[:-1])), key=lambda q: (-row[q], q))
            while remaining:
                anchor = remaining.pop(0)
                group, rest = [anchor], []
                for query in remaining:
                    (group if compatibility[image, query, group].all() else rest).append(query)
                remaining = rest
                labels[image, state, group] = anchor
    indices = torch.as_tensor(labels, device=logits.device)
    # One batched segmented logsumexp replaces up to Q tiny GPU reductions.
    maxima = torch.full_like(per_image, -torch.inf).scatter_reduce(
        -1, indices, per_image.detach(), reduce="amax", include_self=True)
    shifted = per_image - maxima.gather(-1, indices).masked_fill(~torch.isfinite(per_image), 0)
    masses = torch.zeros_like(per_image).scatter_add(-1, indices, shifted.exp())
    grouped = maxima + masses.clamp_min(torch.finfo(masses.dtype).tiny).log()
    # Singleton groups keep exactly the original value, including -inf and EOF.
    counts = torch.zeros_like(per_image).scatter_add(-1, indices, torch.ones_like(per_image))
    return torch.where(counts == 1, per_image, grouped).reshape_as(logits)
