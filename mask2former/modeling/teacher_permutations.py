"""Best-PQ teacher selection with sample-major candidate ordering."""
import torch


def generate_permutations(indices, k, device):
    if k < 1:
        raise ValueError('At least one teacher candidate is required')
    return [src.to(device)[torch.randperm(len(src), device=device)].tolist()
            for _ in range(k) for src, _ in indices]


def select_best_permutations(orders, pq):
    if pq.ndim != 2 or len(orders) != pq.numel():
        raise ValueError('PQ must have shape [K, batch] matching the candidate orders')
    if not torch.isfinite(pq).all():
        raise ValueError('Teacher candidate PQ must be finite')
    batch = pq.shape[1]
    return [orders[winner * batch + image]
            for image, winner in enumerate(pq.argmax(0).tolist())]
