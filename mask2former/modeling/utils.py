"""Utilities for mask losses that align high-resolution targets with low-resolution logits."""

from typing import Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_mask_block_counts(
    targets: torch.Tensor, size: Tuple[int, int]
) -> Tuple[torch.Tensor, int, int, int]:
    """Aggregate target mask positives for each low-resolution prediction block.

    Args:
        targets: Tensor of shape (N, H_t, W_t) containing binary mask targets.
        size: Tuple with the desired spatial resolution (h, w).

    Returns:
        A tuple ``(pos_counts, block_area, H_t, W_t)`` where ``pos_counts`` has shape
        ``(N, h * w)`` with the number of positive pixels for each block,
        ``block_area`` is the number of high-resolution pixels represented by a
        single low-resolution logit, and ``H_t``/``W_t`` are the original target
        dimensions.
    """

    if targets.ndim != 3:
        raise ValueError(
            "Expected targets with shape (N, H, W); got "
            f"{tuple(targets.shape)} instead"
        )

    n, H_t, W_t = targets.shape
    h, w = size

    if H_t == h and W_t == w:
        pos_counts = targets.reshape(n, h * w).to(dtype=torch.float32)
        return pos_counts, 1, H_t, W_t

    if H_t % h != 0 or W_t % w != 0:
        raise ValueError(
            f"Target resolution {(H_t, W_t)} is not divisible by desired size {(h, w)}"
        )

    step_h = H_t // h
    step_w = W_t // w
    block_area = step_h * step_w

    if n == 0:
        empty = targets.new_zeros((0, h * w), dtype=torch.float32)
        return empty, block_area, H_t, W_t

    targets_float = targets.unsqueeze(1).to(dtype=torch.float32)
    unfolded = F.unfold(targets_float, kernel_size=(step_h, step_w), stride=(step_h, step_w))
    pos_counts = unfolded.sum(dim=1)

    return pos_counts, block_area, H_t, W_t
