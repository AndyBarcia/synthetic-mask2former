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


class SampledTrajectory(list):
    """Actions together with the exact detached grouping used to sample them."""

    def __init__(self, actions, group_indices):
        super().__init__(actions)
        self.group_indices = group_indices


def group_logits(logits, compatibility, *, group_indices=None, return_indices=False):
    """Log-sum-exp each group at its highest-logit representative's index.

    Groups are built greedily in descending individual-logit order, with query
    index breaking ties. Every pair in a group must be compatible (no chains).
    EOF is its own action. Group topology is detached; probability sums remain
    differentiable. Supports [B,Q+1] and [B,T,Q+1] logits.

    return_indices records the sampled partition and its representatives.
    Passing group_indices replays that partition with current differentiable
    logits, avoiding representative or membership changes during rescoring.
    """
    batch, queries = compatibility.shape[:2]
    if logits.shape[0] != batch or logits.shape[-1] != queries + 1:
        raise ValueError("Grouped logits and mask compatibility shapes differ")
    # Most images have no collapsible pair. Their logits already are grouped.
    off_diagonal = compatibility.copy()
    diagonal = np.arange(queries)
    off_diagonal[:, diagonal, diagonal] = False
    if group_indices is None and not off_diagonal.any():
        if return_indices:
            indices = np.broadcast_to(np.arange(queries + 1), logits.shape).copy()
            return logits, indices
        return logits
    per_image = logits.reshape(batch, -1, queries + 1)
    # Transfer topology once, rather than once per state and once per group.
    raw = per_image.detach().float().cpu().numpy() if group_indices is None else None
    labels = (np.broadcast_to(np.arange(queries + 1), raw.shape).copy()
              if group_indices is None else np.asarray(group_indices).reshape(per_image.shape))
    for image in range(batch) if group_indices is None else ():
        for state, row in enumerate(raw[image]):
            remaining = np.asarray(sorted(np.flatnonzero(np.isfinite(row[:-1])),
                                          key=lambda q: (-row[q], q)), dtype=np.int64)
            while remaining.size:
                anchor, rest = remaining[0], remaining[1:]
                # Only candidates compatible with the anchor can join. Update
                # their eligibility in bulk as each complete-link member joins.
                allowed = compatibility[image, rest, anchor].copy()
                selected = np.zeros(rest.size, dtype=bool)
                for index in np.flatnonzero(allowed):
                    if allowed[index]:
                        selected[index] = True
                        allowed &= compatibility[image, rest, rest[index]]
                labels[image, state, rest[selected]] = anchor
                remaining = rest[~selected]
    indices = torch.as_tensor(labels, device=logits.device)
    # One batched segmented logsumexp replaces up to Q tiny GPU reductions.
    maxima = torch.full_like(per_image, -torch.inf).scatter_reduce(
        -1, indices, per_image.detach(), reduce="amax", include_self=True)
    shifted = per_image - maxima.gather(-1, indices).masked_fill(~torch.isfinite(per_image), 0)
    masses = torch.zeros_like(per_image).scatter_add(-1, indices, shifted.exp())
    grouped = maxima + masses.clamp_min(torch.finfo(masses.dtype).tiny).log()
    # Singleton groups keep exactly the original value, including -inf and EOF.
    counts = torch.zeros_like(per_image).scatter_add(-1, indices, torch.ones_like(per_image))
    result = torch.where(counts == 1, per_image, grouped).reshape_as(logits)
    return (result, labels.reshape(logits.shape)) if return_indices else result
