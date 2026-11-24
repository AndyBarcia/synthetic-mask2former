"""Dynamic mask assignment selection and training integration."""
import unittest
from unittest.mock import patch
import torch
from detectron2.utils.events import EventStorage
from mask2former.modeling.matcher import dynamic_k_assignment, pairwise_binary_mask_iou
from test_multiple_proposals import criterion


class DynamicKTests(unittest.TestCase):
    def test_iou_controls_count_and_cost_controls_selection(self):
        costs = torch.tensor([[0., 8.], [1., 7.], [8., 0.], [9., 1.]])
        ious = torch.tensor([[.9, .0], [.8, .0], [.7, .6], [.0, .5]])
        src, dst = dynamic_k_assignment(costs, ious, 3)
        self.assertEqual(src.tolist(), [0, 1, 2])
        self.assertEqual(dst.tolist(), [0, 0, 1])
        self.assertEqual(dynamic_k_assignment(costs, ious, 1)[1].tolist(), [0, 1])

    def test_conflicts_choose_lowest_cost_and_can_omit_gt(self):
        costs = torch.tensor([[0., 1.], [4., 5.]])
        src, dst = dynamic_k_assignment(costs, torch.zeros_like(costs))
        self.assertEqual(src.tolist(), [0])
        self.assertEqual(dst.tolist(), [0])

    def test_empty_and_overflow(self):
        for shape in [(0, 2), (3, 0)]:
            src, dst = dynamic_k_assignment(torch.zeros(shape), torch.zeros(shape))
            self.assertEqual(src.numel(), 0)
            self.assertEqual(dst.numel(), 0)
        src, dst = dynamic_k_assignment(torch.rand(2, 5), torch.ones(2, 5))
        self.assertEqual(len(src.unique()), len(src))
        self.assertLessEqual(len(src), 2)
        with self.assertRaises(ValueError):
            dynamic_k_assignment(torch.zeros(2, 2), torch.zeros(2, 2), 0)

    def test_mask_iou_is_binary_and_empty_masks_not_positive(self):
        logits = torch.tensor([[[9., -9.], [9., -9.]], [[-9., -9.], [-9., -9.]]])
        masks = torch.tensor([[[1., 0.], [1., 0.]], [[0., 0.], [0., 0.]]])
        torch.testing.assert_close(pairwise_binary_mask_iou(logits, masks), torch.tensor([[1., 0.], [0., 0.]]))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            dense = pairwise_binary_mask_iou(torch.ones(1, 2, 2), torch.ones(1, 512, 512))
        self.assertEqual(dense.dtype, torch.float32)
        self.assertEqual(dense.item(), 1.)

    def test_training_original_gt_auxiliary_and_gradients(self):
        for kind in ['point', 'block']:
            with self.subTest(kind=kind), EventStorage():
                c = criterion(kind, copies=1)
                c.dynamic_assignment = True
                c.matcher.proposal_assignment = 'dynamic_k'
                c.matcher.dynamic_k_topk = 10
                gt = torch.zeros(2, 8, 8);gt[0, :4] = 1;gt[1, 4:] = 1
                targets = [dict(labels=torch.tensor([1, 2]), masks=gt)]
                logits = torch.tensor([[[0., 8., 0., -5.], [0., 8., 0., -5.], [0., 0., 8., -5.]]], requires_grad=True)
                masks = torch.tensor([[[[9., 9.], [-9., -9.]], [[9., 9.], [-9., -9.]], [[-9., -9.], [9., 9.]]]], requires_grad=True)
                outputs = dict(pred_logits=logits, pred_masks=masks)
                outputs['aux_outputs'] = [dict(pred_logits=logits.clone(), pred_masks=masks.clone())]
                captured = []
                def teacher(out, indices, original):
                    self.assertIs(original, targets)
                    captured.extend(indices)
                    return {'loss_object_decoder': out['pred_logits'].sum() * 0}
                with patch.object(c, 'loss_object_decoder', side_effect=teacher):
                    losses = c(outputs, targets)
                self.assertEqual(captured[0][1].tolist(), [0, 0, 1])
                sum(losses.values()).backward()
                self.assertTrue(torch.isfinite(masks.grad).all())
                self.assertGreater(masks.grad.abs().sum(), 0)
                self.assertGreater(logits.grad.abs().sum(), 0)


if __name__ == '__main__':
    unittest.main()
