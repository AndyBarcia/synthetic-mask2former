"""Compatibility for Detectron2 versions using the legacy CUDA AMP API."""

import torch


def cuda_autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
    """Preserve the legacy argument order while using the current AMP API."""
    return torch.amp.autocast(
        "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
    )


def install_detectron2_amp_compat():
    # AMPTrainer imports this alias inside run_step, so updating a Detectron2
    # module global would not fix it. Keep the legacy signature for callers.
    torch.cuda.amp.autocast = cuda_autocast
