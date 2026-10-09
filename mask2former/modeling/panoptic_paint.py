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
