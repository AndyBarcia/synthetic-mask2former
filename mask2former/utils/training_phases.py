"""Configuration and checkpoint handoff for supervised and decoder RL phases."""
import os


MODES = ("supervised", "supervised_then_rl", "rl", "hybrid", "supervised_then_hybrid")


def validate_training_mode(cfg):
    mode = cfg.TRAINING.MODE
    if mode not in MODES:
        raise ValueError(f"TRAINING.MODE must be one of {MODES}, got {mode!r}")
    if mode == "supervised" and (cfg.MODEL.MASK_FORMER.OBJECT_RL_ONLY or
                                 cfg.MODEL.MASK_FORMER.OBJECT_RL_WEIGHT != 0):
        raise ValueError("Use TRAINING.MODE rl or supervised_then_rl to enable RL")
    if mode != "supervised":
        if cfg.TRAINING.RL.WEIGHT <= 0 or cfg.TRAINING.RL.MAX_ITER <= 0:
            raise ValueError("RL weight and iteration budget must be positive")
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME != "MultiScaleMaskedTransformerDecoder":
            raise ValueError("RL training requires the object decoder")
        if mode in ("rl", "hybrid") and not cfg.MODEL.WEIGHTS:
            raise ValueError("TRAINING.MODE rl/hybrid requires MODEL.WEIGHTS")
    return mode


def phase_config(cfg, phase, weights=None):
    """Clone config; each phase owns an independent solver and output directory."""
    result = cfg.clone()
    result.defrost()
    rl = phase in ("rl", "hybrid")
    # Selected-map supervision applies only to hybrid training.
    if phase != "hybrid":
        result.MODEL.MASK_FORMER.HYBRID_SELECTED_PANOPTIC_SUPERVISION = False
    result.MODEL.MASK_FORMER.OBJECT_RL_ONLY = phase == "rl"
    result.MODEL.MASK_FORMER.OBJECT_RL_WEIGHT = cfg.TRAINING.RL.WEIGHT if rl else 0.0
    if cfg.TRAINING.MODE in ("supervised_then_rl", "supervised_then_hybrid"):
        result.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, phase)
    if weights is not None:
        result.MODEL.WEIGHTS = weights
    if rl:
        for key in ("MAX_ITER", "BASE_LR", "WARMUP_ITERS", "STEPS", "LR_SCHEDULER_NAME"):
            setattr(result.SOLVER, key, getattr(cfg.TRAINING.RL, key))
    result.freeze()
    return result


def evaluation_config(cfg):
    """Disable training objectives without changing checkpoint/output paths."""
    result = cfg.clone()
    result.defrost()
    result.MODEL.MASK_FORMER.OBJECT_RL_ONLY = False
    result.MODEL.MASK_FORMER.OBJECT_RL_WEIGHT = 0.0
    result.MODEL.MASK_FORMER.HYBRID_SELECTED_PANOPTIC_SUPERVISION = False
    result.freeze()
    return result
