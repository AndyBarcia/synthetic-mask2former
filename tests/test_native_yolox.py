"""Native model contract, gradients, inference, and checkpoint parity."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Instances

from mask2former.yolox_model import YOLOXObjectDecoder, ImageMemories
from mask2former.modeling.yolox.proposals import YOLOXMaskProposals
from mask2former.modeling.yolox.mask_first import MaskFirstProposals
from mask2former.modeling.yolox.policy import DynamicMaskPolicy, ProposalObjectDecoder


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


def model(box_free=False, frozen_pixel=False):
    head = MaskFirstProposals(256, 4, {0, 2}, 16) if box_free else YOLOXMaskProposals(256, 4, {0, 2}, 16, dynamic_masks=True)
    decoder = ProposalObjectDecoder(256, 32, 20, 1, 4, 64, prefix_tree_branches=2)
    return YOLOXObjectDecoder(
        backbone=Backbone(), pixel_decoder=Pixel(), proposal_head=head,
        memory=ImageMemories(256, 32), policy=DynamicMaskPolicy(decoder, 4, 256),
        metadata=SimpleNamespace(thing_dataset_id_to_contiguous_id={1: 0, 2: 2}),
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1], freeze_backbone=True,
        freeze_pixel_decoder=frozen_pixel, freeze_memory=True, freeze_proposals=False,
        budget=20, max_steps=8, rendering='forward', semantic_on=True, instance_on=True)


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

    def test_pixel_gradients_and_frozen_backbone_for_both_heads(self):
        for box_free in (False, True):
            m = model(box_free).train()
            losses = m(batch())
            self.assertEqual(set(losses), {'loss_proposal', 'loss_object_decoder'})
            sum(losses.values()).backward()
            self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
            self.assertGreater(float(m.sem_seg_head.pixel_decoder.conv.weight.grad.norm()), 0)
            for name, p in m.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
            m.zero_grad(set_to_none=True)
            sum(m(batch(empty=True)).values()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_sequence_loss_reaches_trainable_image_memories_and_backbone(self):
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
        masks = torch.ones(2, 20, 4, 4)
        excluded = m.policy.decoder.prepare_vocabulary_exclusions(masks)
        self.assertTrue(torch.equal(excluded[0], torch.eye(20, dtype=torch.bool)))
        sum(m(batch()).values()).backward()
        self.assertTrue(all(p.grad is None for p in m.sem_seg_head.pixel_decoder.parameters()))

    def test_detectron_checkpoint_roundtrip_and_prediction_contract(self):
        m = model().eval()
        inputs = batch()
        with torch.no_grad():
            expected = m(inputs)[0]
        self.assertEqual(tuple(expected['panoptic_seg'][0].shape), (71, 79))
        self.assertEqual(tuple(expected['sem_seg'].shape), (4, 71, 79))
        self.assertEqual(expected['instances'].image_size, (71, 79))
        with tempfile.TemporaryDirectory() as folder:
            DetectionCheckpointer(m, folder).save('model')
            restored = model().eval()
            DetectionCheckpointer(restored).load(str(Path(folder) / 'model.pth'))
            with torch.no_grad():
                actual = restored(inputs)[0]
            self.assertTrue(torch.equal(actual['panoptic_seg'][0], expected['panoptic_seg'][0]))
            self.assertEqual(actual['panoptic_seg'][1], expected['panoptic_seg'][1])
            for key, value in m.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)

    def test_square_mapper_and_prediction_padding_agree(self):
        import numpy as np
        from PIL import Image
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.data import transforms as T
        from mask2former import add_maskformer2_config
        from mask2former.data.dataset_mappers.yolox_panoptic_test_mapper import YOLOXPanopticTestMapper
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.MODEL.YOLOX.PAD_SIZE = 64
        cfg.INPUT.FORMAT = 'RGB'
        image = np.arange(29 * 47 * 3, dtype=np.uint8).reshape(29, 47, 3)
        with tempfile.TemporaryDirectory() as folder:
            file = str(Path(folder) / 'image.png')
            Image.fromarray(image).save(file)
            mapped = YOLOXPanopticTestMapper(cfg)({'file_name': file})
        resized = T.ResizeScale(1., 1., 64, 64).get_transform(image).apply_image(image)
        expected = T.FixedSizeCrop((64, 64)).get_transform(resized).apply_image(resized)
        self.assertTrue(np.array_equal(mapped['image'].numpy().transpose(1, 2, 0), expected))
        m = model().eval()
        m.pad_size = 64
        unpadded = dict(mapped, image=torch.from_numpy(resized.transpose(2, 0, 1).copy()))
        unpadded.pop('valid_resized_size')
        with torch.no_grad():
            a, b = m([mapped])[0], m([unpadded])[0]
        self.assertTrue(torch.equal(a['panoptic_seg'][0], b['panoptic_seg'][0]))
        self.assertEqual(a['panoptic_seg'][1], b['panoptic_seg'][1])

    def test_native_decoder_only_rl_keeps_features_frozen(self):
        m = model(frozen_pixel=True)
        m.freeze_proposals = True
        m.sem_seg_head.proposal_head.requires_grad_(False)
        m.rl_only, m.rl_weight, m.rl_samples, m.max_steps = True, 1., 2, 3
        m.train()
        losses = m(batch())
        self.assertEqual(set(losses), {'loss_object_rl'})
        self.assertTrue(bool(torch.isfinite(losses['loss_object_rl'])))
        losses['loss_object_rl'].backward()
        self.assertTrue(any(p.grad is not None for p in m.policy.parameters()))
        self.assertTrue(all(p.grad is None for p in m.sem_seg_head.parameters()))

    def test_production_head_matches_research_head(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        if not (Path(__file__).resolve().parents[1] / 'tools/yolox_mask_proposals.py').exists():
            self.skipTest('Optional historical research implementation is not tracked on this branch')
        from yolox_mask_proposals import YOLOXMaskProposals as ResearchHead
        old = ResearchHead(256, 4, {0, 2}, 16, dynamic_masks=True)
        new = YOLOXMaskProposals(256, 4, {0, 2}, 16, dynamic_masks=True)
        new.load_state_dict(old.state_dict(), strict=True)
        dense = torch.randn(1, 256, 16, 16)
        multi = [torch.randn(1, 256, n, n) for n in (2, 4, 8)]
        a = old.proposals(old(multi, dense, (64, 64)), dense, topk=20, nms=None)
        b = new.proposals(new(multi, dense, (64, 64)), dense, topk=20)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
