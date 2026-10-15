import os
import runpy

from detectron2.config import get_cfg
from mask2former.config import add_maskformer2_config

phases = runpy.run_path(os.path.join(os.path.dirname(__file__), '../mask2former/utils/training_phases.py'))


def test_hybrid_keeps_supervision_and_uses_rl_schedule():
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.TRAINING.MODE = 'hybrid'
    cfg.MODEL.WEIGHTS = 'pretrained.pth'
    cfg.freeze()
    assert phases['validate_training_mode'](cfg) == 'hybrid'
    hybrid = phases['phase_config'](cfg, 'hybrid')
    assert not hybrid.MODEL.MASK_FORMER.OBJECT_RL_ONLY
    assert hybrid.MODEL.MASK_FORMER.OBJECT_RL_WEIGHT == cfg.TRAINING.RL.WEIGHT
    assert hybrid.SOLVER.MAX_ITER == cfg.TRAINING.RL.MAX_ITER
    assert hybrid.MODEL.WEIGHTS == 'pretrained.pth'
    assert cfg.MODEL.MASK_FORMER.OBJECT_RL_WEIGHT == 0
    assert phases['phase_config'](cfg, 'rl').MODEL.MASK_FORMER.OBJECT_RL_ONLY


def test_sequential_hybrid_checkpoint_handoff():
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    cfg.TRAINING.MODE = 'supervised_then_hybrid'
    cfg.OUTPUT_DIR = 'output/example'
    supervised = phases['phase_config'](cfg, 'supervised')
    hybrid = phases['phase_config'](cfg, 'hybrid', 'output/example/supervised/model_final.pth')
    assert supervised.OUTPUT_DIR == 'output/example/supervised'
    assert supervised.MODEL.MASK_FORMER.OBJECT_RL_WEIGHT == 0
    assert hybrid.OUTPUT_DIR == 'output/example/hybrid'
    assert not hybrid.MODEL.MASK_FORMER.OBJECT_RL_ONLY
    assert hybrid.MODEL.WEIGHTS == 'output/example/supervised/model_final.pth'
