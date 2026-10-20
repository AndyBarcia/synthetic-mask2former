"""Call-local, differentiable memoization of exact discrete map states."""
from dataclasses import dataclass

import torch


@dataclass
class StateLookup:
    keys: torch.Tensor
    representatives: torch.Tensor
    inverse: torch.Tensor
    missing: torch.Tensor
    old_size: int


class StateCache:
    """Hash-sort candidate states; verify full keys before sharing any value.

    Fingerprints only accelerate grouping. Even deliberate hash collisions
    cannot merge unequal keys (they can merely reduce the number of hits).
    Values retain autograd history; the owning policy call controls lifetime.
    """
    def __init__(self):
        self.keys = None
        self.values = None
        self.weights = None
        self.evaluated = 0
        self.hits = 0

    def store_unique(self, keys, values):
        """Append keys proved distinct from each other and every cached key."""
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = torch.cat((self.keys, keys))
            self.values = torch.cat((self.values, values))
        self.evaluated += len(keys)
        return values

    def lookup(self, keys):
        old_size = 0 if self.keys is None else len(self.keys)
        combined = keys if self.keys is None else torch.cat((self.keys, keys))
        if self.weights is None:
            generator = torch.Generator(device=keys.device).manual_seed(982451653)
            self.weights = torch.randint(1, 2**62, (keys.shape[1],),
                                         generator=generator, device=keys.device)
        fingerprints = (combined.long() * self.weights).sum(-1)
        order = fingerprints.argsort(stable=True)
        ordered = combined[order]
        equal = ((fingerprints[order[1:]] == fingerprints[order[:-1]]) &
                 (ordered[1:] == ordered[:-1]).all(-1))
        first = torch.cat((equal.new_ones(1), ~equal))
        representatives = order[first]
        groups = first.long().cumsum(0) - 1
        inverse = torch.empty_like(order).scatter_(0, order, groups)
        missing = representatives[representatives >= old_size] - old_size
        return StateLookup(combined, representatives, inverse[old_size:], missing, old_size)

    def store(self, lookup, values):
        source = values if self.values is None else (
            self.values if values is None else torch.cat((self.values, values)))
        is_new = lookup.representatives >= lookup.old_size
        new_index = is_new.long().cumsum(0) - 1 + lookup.old_size
        indices = torch.where(is_new, new_index, lookup.representatives)
        self.keys = lookup.keys[lookup.representatives]
        self.values = source[indices]
        self.evaluated += len(lookup.missing)
        self.hits += len(lookup.inverse) - len(lookup.missing)
        return self.values[lookup.inverse]


def pack_by_image(images, batch_size):
    """Assign dense per-image slots without a Python loop over images."""
    counts = images.bincount(minlength=batch_size)
    capacity = int(counts.max())
    order = images.argsort(stable=True)
    offsets = counts.cumsum(0) - counts
    slots = torch.empty_like(images)
    slots[order] = torch.arange(len(images), device=images.device) - offsets[images[order]]
    return slots, capacity
