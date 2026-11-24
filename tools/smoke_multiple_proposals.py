"""GPU forward/backward, inference, and decoder-only RL for K-proposal training."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from detectron2.utils.events import EventStorage
from train_net import get_cfg, add_deeplab_config, add_maskformer2_config, Trainer
from mask2former.data.synthetic_scene_batch_loader import SyntheticSceneBatchLoader

p = argparse.ArgumentParser()
p.add_argument('--config-file', default=str(Path(__file__).resolve().parents[1] / 'configs/synthetic-scene/panoptic-segmentation/maskseq_R50_bs16_10kpre_2krl_multiple_proposals.yaml'))
a = p.parse_args()
torch.set_num_threads(2); torch.manual_seed(42)
cfg = get_cfg(); add_deeplab_config(cfg); add_maskformer2_config(cfg)
cfg.merge_from_file(a.config_file); cfg.SOLVER.IMS_PER_BATCH = 2
model = Trainer.build_model(cfg).cuda()
batch = next(iter(SyntheticSceneBatchLoader(cfg, True)))
with EventStorage():
    model.train()
    with torch.autocast('cuda', dtype=torch.float16):
        losses = model(batch)
    assert all(torch.isfinite(v) for v in losses.values()), losses
    assert len([k for k in losses if k.startswith('loss_mask')]) == 6
    assert not any('query_bias' in k for k in losses)
    sum(losses.values()).backward()
    assert model.sem_seg_head.predictor.class_embed.weight.grad.abs().sum() > 0
    assert model.criterion.object_decoder.bos.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.sem_seg_head.predictor.query_bias_embed.parameters())
    print('Six-layer point-sampled K-proposal supervision and best-of-K teacher backward passed', flush=True)
    model.zero_grad(set_to_none=True); model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        predictions = model(batch)
    assert all(p['panoptic_seg'][0].shape == (record['height'], record['width'])
               for p, record in zip(predictions, batch))
    print('Whole-object panoptic and instance inference passed', flush=True)
    model.train(); model.object_rl_only = True
    model.backbone.requires_grad_(False); model.sem_seg_head.requires_grad_(False)
    model.criterion.object_rl_weight = 1.; model.criterion.object_rl_num_samples = 2
    model.criterion.weight_dict['loss_object_rl'] = 1.
    with torch.autocast('cuda', dtype=torch.float16):
        losses = model(batch)
    assert list(losses) == ['loss_object_rl'] and torch.isfinite(losses['loss_object_rl'])
    losses['loss_object_rl'].backward()
    assert model.criterion.object_decoder.bos.grad is not None
    assert all(p.grad is None for p in model.backbone.parameters())
    assert 'mask2former.modeling.full_mask_losses' not in sys.modules
    print('Decoder-only RL and independence from full-resolution losses passed', flush=True)
print('ALL MULTIPLE-PROPOSAL GPU CHECKS PASSED', flush=True)
