"""YOLOX head/criterion/model contracts through Detectron2 interfaces."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances

from mask2former.yolox_model import YOLOXObjectDecoder
from mask2former.modeling.meta_arch.yolox_head import YOLOXHead, ImageMemories, ProposalTokenAdapter
from mask2former.modeling.yolox.proposals import YOLOXMaskProposals
from mask2former.modeling.yolox.object_decoder import ProposalObjectDecoder
from mask2former.modeling.yolox.matcher import YOLOXMatcher
from mask2former.modeling.yolox.criterion import YOLOXCriterion


class Backbone(nn.Module):
    size_divisibility = 32

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, 4, stride=4)

    def forward(self, images):
        return self.conv(images)


class Pixel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 256, 1)

    def forward_features(self, features):
        dense = self.conv(features)
        return dense, None, [F.adaptive_avg_pool2d(dense, n) for n in (2, 4, 8)]


def model(box_free=False, frozen_pixel=False, rl=False):
    head = YOLOXHead(pixel_decoder=Pixel(), predictor=YOLOXMaskProposals(
        256, 4, {0, 2}, 16, box_free=box_free),
        memory=ImageMemories(256, 32), token_adapter=ProposalTokenAdapter(4, 256), num_classes=4, num_queries=86)
    decoder = ProposalObjectDecoder(256, 32, 86, 1, 4, 64, prefix_tree_branches=2)
    weights = {name: 1. for name in ['loss_objectness', 'loss_ce', 'loss_box', 'loss_mask', 'loss_dice',
                                   'loss_stuff_presence', 'loss_stuff_mask', 'loss_stuff_dice',
                                   'loss_object_decoder', 'loss_object_rl']}
    criterion = YOLOXCriterion(4, YOLOXMatcher(4, {0, 2}, box_free=box_free), weights, decoder, {0, 2},
                               object_rl_weight=1. if rl else 0., object_rl_only=rl,
                               object_rl_num_samples=2, object_rl_max_steps=3)
    return YOLOXObjectDecoder(
        backbone=Backbone(), sem_seg_head=head, criterion=criterion,
        metadata=SimpleNamespace(thing_dataset_id_to_contiguous_id={1: 0, 2: 2}),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1], freeze_backbone=True,
        freeze_pixel_decoder=frozen_pixel, freeze_memory=True, freeze_proposals=False,
        num_queries=86, max_steps=8, semantic_on=True, instance_on=True)


def batch(empty=False):
    masks = torch.zeros(0 if empty else 2, 59, 61, dtype=torch.bool)
    if not empty:
        masks[0, 8:35, 8:35] = True
        masks[1, 35:] = True
    gt = Instances((59, 61))
    gt.gt_masks = masks
    gt.gt_classes = torch.tensor([] if empty else [0, 1], dtype=torch.long)
    return [dict(image=torch.randn(3, 59, 61), instances=gt, height=71, width=79)]


class NativeYOLOXTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_oracle_receives_full_resized_proposal_pool(self):
        m = model().eval()
        inputs = batch()
        inputs[0]['panoptic_ground_truth'] = (torch.zeros(71, 79), [])
        captured = {}
        handle = m.sem_seg_head.register_forward_hook(lambda module, args, out: captured.update(out))
        with torch.no_grad():
            result = m(inputs)[0]
        handle.remove()
        expected = F.interpolate(captured['proposals'][0]['masks'][None].float(),
                                 (64, 64), mode='bilinear', align_corners=False)[0, :, :59, :61]
        expected = F.interpolate(expected[None], (71, 79), mode='bilinear', align_corners=False)[0]
        self.assertEqual(tuple(result['panoptic_proposals'].shape), (86, 71, 79))
        torch.testing.assert_close(result['panoptic_proposals'], expected)

    def test_named_criterion_losses_and_gradients_for_both_heads(self):
        for box_free in (False, True):
            m = model(box_free).train()
            losses = m(batch())
            self.assertIn('loss_object_decoder', losses)
            self.assertIn('loss_mask', losses)
            self.assertEqual('loss_box' in losses, not box_free)
            sum(losses.values()).backward()
            self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
            for name, p in m.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
            m.zero_grad(set_to_none=True)
            sum(m(batch(empty=True)).values()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_sequence_loss_reaches_image_memories_and_backbone(self):
        m = model()
        m.freeze_backbone = m.freeze_memory = False
        m.backbone.requires_grad_(True)
        m.sem_seg_head.memory.requires_grad_(True)
        m.train()
        m(batch())['loss_object_decoder'].backward()
        self.assertGreater(float(m.backbone.conv.weight.grad.norm()), 0)
        self.assertGreater(float(m.sem_seg_head.pixel_decoder.conv.weight.grad.norm()), 0)
        self.assertGreater(float(m.sem_seg_head.memory.level_embed.weight.grad.norm()), 0)

    def test_freeze_modes_and_exact_id_exclusions(self):
        m = model(frozen_pixel=True).train()
        self.assertFalse(m.backbone.training)
        self.assertFalse(m.sem_seg_head.pixel_decoder.training)
        self.assertFalse(m.sem_seg_head.memory.training)
        excluded = m.criterion.object_decoder.prepare_vocabulary_exclusions(torch.ones(2, 20, 4, 4))
        self.assertTrue(torch.equal(excluded[0], torch.eye(20, dtype=torch.bool)))
        sum(m(batch()).values()).backward()
        self.assertTrue(all(p.grad is None for p in m.sem_seg_head.pixel_decoder.parameters()))

    def test_detectron_checkpoint_roundtrip_and_prediction_contract(self):
        m, inputs = model().eval(), batch()
        with torch.no_grad():
            expected = m(inputs)[0]
        self.assertEqual(tuple(expected['panoptic_seg'][0].shape), (71, 79))
        self.assertEqual(tuple(expected['sem_seg'].shape), (4, 71, 79))
        self.assertEqual(expected['instances'].image_size, (71, 79))
        self.assertTrue(any(k.startswith('criterion.object_decoder.') for k in m.state_dict()))
        self.assertFalse(any(k.startswith('policy.') for k in m.state_dict()))
        with tempfile.TemporaryDirectory() as folder:
            DetectionCheckpointer(m, folder).save('model')
            restored = model().eval()
            DetectionCheckpointer(restored).load(str(Path(folder) / 'model.pth'))
            with torch.no_grad():
                actual = restored(inputs)[0]
            self.assertTrue(torch.equal(actual['panoptic_seg'][0], expected['panoptic_seg'][0]))
            self.assertEqual(actual['panoptic_seg'][1], expected['panoptic_seg'][1])

    def test_decoder_only_rl_keeps_head_frozen_and_adapter_trainable(self):
        m = model(rl=True).train()
        losses = m(batch())
        self.assertEqual(set(losses), {'loss_object_rl'})
        losses['loss_object_rl'].backward()
        self.assertTrue(any(p.grad is not None for p in m.criterion.object_decoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.sem_seg_head.token_adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in m.sem_seg_head.predictor.parameters()))
        self.assertTrue(all(p.grad is None for p in m.sem_seg_head.pixel_decoder.parameters()))


if __name__ == '__main__':
    unittest.main()
