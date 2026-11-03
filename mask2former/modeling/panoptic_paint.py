"""Batched last-mask-wins painting with stuff merging and compact segment IDs."""
import torch


@torch.no_grad()
def paint_panoptic_masks(logits, labels, valid, isthing, reverse=False):
    """Inputs B,Q,H,W / B,Q; return maps and GPU class tables (void=-1).

    Empty masks allocate no IDs; later masks overwrite earlier masks. Repeated
    stuff classes share their first segment ID. Fully occluded slots are padding.
    """
    batch, queries, height, width = logits.shape
    if queries == 0:
        return (torch.zeros((batch, height, width), device=logits.device, dtype=torch.int32),
                torch.full((batch, 1), -1, device=logits.device, dtype=torch.int32))
    if reverse:
        logits, labels, valid = logits.flip(1), labels.flip(1), valid.flip(1)
    masks = (logits > 0) & valid[:, :, None, None]
    nonempty = masks.flatten(2).any(-1)
    things = isthing[labels]
    # Record the first nonempty occurrence of each stuff class in paint order.
    classes = isthing.numel()
    occurrences = torch.nn.functional.one_hot(labels, classes).bool()
    occurrences &= (nonempty & ~things)[:, :, None]
    first = occurrences.cumsum(1).gather(2, labels[:, :, None]).squeeze(2) == 1
    new_segment = nonempty & (things | first)
    segment_ids = new_segment.cumsum(1)
    first_stuff = torch.full((batch, classes), queries + 1, device=logits.device, dtype=torch.long)
    first_stuff.scatter_reduce_(1, labels, torch.where(nonempty & ~things, segment_ids, queries + 1),
                                reduce='amin', include_self=True)
    segment_ids = torch.where(things, segment_ids, first_stuff.gather(1, labels))
    segment_ids = segment_ids.masked_fill(~nonempty, 0)
    # Max returns the first True in reversed order, i.e. the last painted mask.
    foreground, winners = masks.flip(1).to(torch.uint8).max(1)
    painted = segment_ids.flip(1).gather(1, winners.flatten(1)).reshape(batch, height, width)
    painted = painted.masked_fill(foreground == 0, 0).to(torch.int32)
    class_table = torch.full((batch, queries + 1), -1, device=logits.device, dtype=torch.int32)
    class_table.scatter_(1, segment_ids, labels.to(torch.int32))
    offsets = torch.arange(batch, device=logits.device)[:, None] * (queries + 1)
    area = torch.bincount((painted.flatten(1).long() + offsets).flatten(),
                          minlength=batch * (queries + 1)).reshape(batch, queries + 1)
    class_table.masked_fill_(area == 0, -1)
    class_table[:, 0] = -1
    return painted, class_table


@torch.no_grad()
def score_panoptic_masks(logits, labels, valid, isthing, scores, overlap_threshold=.8):
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
