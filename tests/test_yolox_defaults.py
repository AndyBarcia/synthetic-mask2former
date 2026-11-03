from pathlib import Path
import unittest
import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from tests.test_native_yolox import model, batch


class YOLOXDefaultsTest(unittest.TestCase):
    def test_default_schema_and_canonical_recipe(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        self.assertEqual(cfg.MODEL.YOLOX.TOPK, 337)
        self.assertEqual(cfg.MODEL.YOLOX.MAX_STEPS, 25)
        for removed in ['TOKEN_FUSION', 'OBJECT_DEC_SUPERVISION', 'DYNAMIC_MASKS',
                        'FULL_MASK_LOSS', 'MATCH_IOU_THRESHOLD', 'RENDERING',
                        'OBJECT_DEC_SAMPLE_SUFFIX_PROPOSALS']:
            self.assertNotIn(removed, cfg.MODEL.YOLOX)
        root = Path(__file__).resolve().parents[1]
        cfg.merge_from_file(str(root / 'configs/synthetic/yolox.yaml'))
        self.assertEqual(cfg.MODEL.META_ARCHITECTURE, 'YOLOXObjectDecoder')
        self.assertEqual(cfg.SOLVER.MAX_ITER, 2000)
        self.assertEqual(cfg.TRAINING.RL.MAX_ITER, 200)
        self.assertEqual(cfg.SOLVER.IMS_PER_BATCH, 8)
        self.assertEqual(cfg.TRAINING.MODE, 'supervised_then_rl')
        self.assertTrue(cfg.TEST.SYNTHETIC_PANOPTIC_ORACLE)

    def test_default_training_covers_gt_and_uses_same_mask_positive_groups(self):
        torch.set_num_threads(1)
        m = model().train()
        captured = {}
        m.sem_seg_head.register_forward_hook(lambda module, args, out: captured.update(out))
        inputs = batch()
        losses = m(inputs)
        from detectron2.structures import ImageList
        images = ImageList.from_tensors([r['image'] for r in inputs], m.size_divisibility)
        targets = m.prepare_targets([r['instances'] for r in inputs], images)
        _, prepared, indices = m.criterion.loss_proposals(captured, targets, return_assignment=True)
        representatives, groups = m.criterion.dynamic_teacher_groups(captured, prepared, indices)
        self.assertEqual(len(representatives[0]), 2)
        ids, gt = indices[0]
        assigned = captured['proposals'][0]['dense_ids'][groups[0][int(representatives[0][0])]]
        self.assertEqual(set(assigned.tolist()), set(ids[gt == 0].tolist()))
        self.assertFalse(hasattr(m.criterion.matcher, 'object_matcher'))
        sum(losses.values()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))


if __name__ == '__main__':
    unittest.main()
