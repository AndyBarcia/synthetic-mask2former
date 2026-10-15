"""Exercise the actual phase driver with tiny checkpointed models and optimizers."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from mask2former.config import add_maskformer2_config
import train_net


def test_driver_hands_off_weights_and_resumes_hybrid_optimizer():
    with tempfile.TemporaryDirectory() as directory:
        cfg = get_cfg()
        add_maskformer2_config(cfg)
        cfg.TRAINING.MODE = 'supervised_then_hybrid'
        cfg.MODEL.WEIGHTS = ''
        cfg.MODEL.MASK_FORMER.HYBRID_SELECTED_PANOPTIC_SUPERVISION = True
        cfg.SOLVER.MAX_ITER = 1
        cfg.TRAINING.RL.MAX_ITER = 1
        cfg.OUTPUT_DIR = directory
        cfg.freeze()
        observations = []
        class TinyTrainer:
            def __init__(self, phase_cfg):
                self.cfg = phase_cfg
                self.phase = Path(phase_cfg.OUTPUT_DIR).name
                self.model = torch.nn.Linear(1, 1, bias=False)
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=phase_cfg.SOLVER.BASE_LR, momentum=.9)
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1)
                self.checkpointer = DetectionCheckpointer(self.model, phase_cfg.OUTPUT_DIR,
                                                         optimizer=self.optimizer, scheduler=self.scheduler)
                self.start_iter = 0
            def resume_or_load(self, resume):
                path = self.cfg.MODEL.WEIGHTS
                restoring = resume and self.checkpointer.has_checkpoint()
                checkpoint = self.checkpointer.resume_or_load(path, resume=resume)
                if restoring:
                    self.start_iter = checkpoint['iteration'] + 1
                observations.append((self.phase, self.start_iter,
                                     bool(self.optimizer.state), self.model.weight.item(),
                                     self.cfg.MODEL.MASK_FORMER.HYBRID_SELECTED_PANOPTIC_SUPERVISION))
            def train(self):
                if self.start_iter >= self.cfg.SOLVER.MAX_ITER:
                    return 'already complete'
                self.optimizer.zero_grad()
                self.model(torch.ones(1, 1)).sum().backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.phase == 'supervised':
                    with torch.no_grad():
                        self.model.weight.fill_(7.)
                self.checkpointer.save('model_final', iteration=self.cfg.SOLVER.MAX_ITER - 1)
                return self.phase
        args = SimpleNamespace(eval_only=False, resume=False)
        with patch.object(train_net, 'setup', return_value=cfg), \
             patch.object(train_net, 'Trainer', TinyTrainer), \
             patch.object(train_net, 'install_detectron2_amp_compat'):
            assert train_net.main(args) == 'hybrid'
            assert [o[0] for o in observations] == ['supervised', 'hybrid']
            assert observations[0][4] is False
            assert observations[1][4] is True
            assert observations[1][1] == 0
            assert observations[1][2] is False  # Fresh hybrid optimizer.
            assert observations[1][3] == 7.  # Loaded supervised weights.
            observations.clear()
            args.resume = True
            assert train_net.main(args) == 'already complete'
            assert len(observations) == 1
            assert observations[0][:3] == ('hybrid', 1, True)
