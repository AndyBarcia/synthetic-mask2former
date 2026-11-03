"""Pixel competition alternative; forward painting remains the default."""
import torch
from ..panoptic_paint import paint_panoptic_masks


@torch.no_grad()
def score_panoptic_masks(logits, labels, valid, isthing, scores, overlap_threshold=0.):
    """Mask2Former pixel competition on the decoder-selected proposals.

    Match the original rule: assigned area counts every argmax pixel, whereas
    original area and emitted pixels use sigmoid(mask) >= .5. Rejected winners
    stay void; there is no second assignment pass.
    """
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in [0, 1]")
    batch, queries, height, width = logits.shape
    if queries == 0:
        return paint_panoptic_masks(logits, labels, valid, isthing)
    probabilities = logits.sigmoid()
    weighted = (probabilities * scores[:, :, None, None]).masked_fill(~valid[:, :, None, None], -torch.inf)
    winners = weighted.argmax(1)
    assigned = winners[:, None] == torch.arange(queries, device=logits.device)[None, :, None, None]
    original = probabilities >= .5
    original_area = original.flatten(2).sum(-1)
    assigned_area = assigned.flatten(2).sum(-1)
    keep = valid & (original_area > 0) & (assigned_area >= overlap_threshold * original_area)
    visible = assigned & original & keep[:, :, None, None]
    # These masks are disjoint, so painting is only used to merge stuff and
    # construct the class table; it cannot introduce any overwriting.
    return paint_panoptic_masks(torch.where(visible, 1., -1.), labels, keep, isthing)
