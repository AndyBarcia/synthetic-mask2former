"""Post-Hungarian IoU filtering, coverage, and loss integration."""
import unittest
from unittest.mock import patch
import torch
from detectron2.utils.events import EventStorage
from mask2former.modeling.multiple_proposals import filter_extra_proposals
from test_multiple_proposals import criterion


def mask_logits(masks):
    return torch.tensor(masks, dtype=torch.float32).mul(20).sub(10)


class FilteredProposalsTests(unittest.TestCase):
    def test_default_tiers_require_third_iou_strictly_above_point_nine(self):
        masks = torch.zeros(1, 3, 10, 10)
        masks[0, 0] = 1
        masks[0, 1].flatten()[:95] = 1
        masks[0, 2].flatten()[:90] = 1
        outputs = dict(pred_masks=masks.mul(20).sub(10),
                       pred_logits=torch.tensor([[[0., 9., 0.]] * 3]))
        target = [dict(masks=torch.ones(1, 10, 10), labels=torch.tensor([1]))]
        indices = [(torch.arange(3), torch.arange(3))]
        def selected():
            return filter_extra_proposals(outputs, target, indices, 3, .5,
                                          True, (.5, .9))[0][0].tolist()
        self.assertEqual(selected(), [0, 1])
        outputs['pred_masks'][0, 2].flatten()[90] = 10
        self.assertEqual(selected(), [0, 1, 2])
        outputs['pred_logits'][0, 2] = torch.tensor([0., 0., 9.])
        self.assertEqual(selected(), [0, 1])

    def test_ranked_thresholds_use_iou_order_and_strict_boundaries(self):
        # Full GT has ten pixels; masks yield exact IoUs .5, .6, .7, .8, .9.
        def run(counts, classes=None):
            masks = torch.zeros(1, 3, 2, 5)
            for q, count in enumerate(counts):
                masks[0, q].flatten()[:count] = 1
            outputs = dict(pred_masks=masks.mul(20).sub(10))
            target = [dict(masks=torch.ones(1, 2, 5), labels=torch.tensor([1]))]
            if classes is not None:
                outputs['pred_logits'] = torch.zeros(1, 3, 4)
                for q, cls in enumerate(classes):
                    outputs['pred_logits'][0, q, cls] = 10
            result = filter_extra_proposals(outputs, target,
                        [(torch.arange(3), torch.arange(3))], 3, .5,
                        classes is not None, (.5, .8))
            return result[0][0].tolist()
        self.assertEqual(run([6, 9, 7]), [1, 2])  # rank by IoU, not query ID
        self.assertEqual(run([9, 8, 8]), [0, 1])  # third is exactly .8
        self.assertEqual(run([9, 9, 9]), [0, 1, 2])
        self.assertEqual(run([9, 5, 5]), [0])     # second is exactly .5
        self.assertEqual(run([0, 0, 0]), [0])     # coverage despite poor masks
        self.assertEqual(run([9, 9, 9], [1, 2, 3]), [0])

    def test_class_filter_rejects_wrong_class_and_no_object_extras(self):
        outputs = dict(pred_masks=torch.full((1, 4, 2, 2), 10.),
                       pred_logits=torch.tensor([[[0., 9., 0., 0.], [0., 0., 9., 0.],
                                                  [0., 0., 0., 9.], [0., 9., 0., 0.]]]))
        target = [dict(masks=torch.ones(1, 2, 2), labels=torch.tensor([1]))]
        indices = [(torch.arange(4), torch.arange(4))]
        self.assertEqual(filter_extra_proposals(outputs, target, indices, 4, .5, True)[0][0].tolist(), [0, 3])
        self.assertEqual(filter_extra_proposals(outputs, target, indices, 4, .5, False)[0][0].tolist(), [0, 1, 2, 3])
        outputs['pred_logits'][0, :, 3] = 20.
        self.assertEqual(filter_extra_proposals(outputs, target, indices, 4, .5, True)[0][0].tolist(), [0])

    def test_strict_threshold_retains_best_and_good_extras(self):
        outputs = dict(pred_masks=mask_logits([[[[0, 0], [0, 0]], [[1, 1], [0, 0]],
                                                [[1, 1], [1, 0]], [[1, 1], [1, 1]]]]))
        targets = [dict(masks=torch.ones(1, 2, 2))]
        indices = [(torch.arange(4), torch.arange(4))]
        src, dst = filter_extra_proposals(outputs, targets, indices, 4, .5)[0]
        self.assertEqual(src.tolist(), [2, 3])
        self.assertEqual(dst.tolist(), [2, 3])
        self.assertEqual(indices[0][0].tolist(), [0, 1, 2, 3])

    def test_every_represented_object_keeps_one_even_if_all_bad(self):
        outputs = dict(pred_masks=torch.full((1, 6, 2, 2), -10.))
        targets = [dict(masks=torch.ones(2, 2, 2))]
        indices = [(torch.arange(6), torch.arange(6))]
        src, dst = filter_extra_proposals(outputs, targets, indices, 3, .5)[0]
        self.assertEqual(src.tolist(), [0, 3])
        self.assertEqual((dst // 3).tolist(), [0, 1])

    def test_empty_and_partial_groups(self):
        outputs = dict(pred_masks=torch.full((2, 2, 2, 2), -10.))
        targets = [dict(masks=torch.ones(3, 2, 2)), dict(masks=torch.empty(0, 2, 2))]
        indices = [(torch.tensor([0, 1]), torch.tensor([4, 8])),
                   (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        result = filter_extra_proposals(outputs, targets, indices, 3, .5)
        self.assertEqual(result[0][1].tolist(), [4, 8])
        self.assertEqual(result[1][0].numel(), 0)

    def test_filtered_losses_auxiliary_normalization_and_no_object_gradients(self):
        for kind in ['point', 'block']:
            with self.subTest(kind=kind), EventStorage():
                c = criterion(kind, copies=3);c.extra_proposal_min_iou = .5
                c.extra_proposal_iou_thresholds = (.5, .9)
                gt = torch.tensor([[[1., 1.], [0., 0.]], [[0., 0.], [1., 1.]]])
                target = [dict(labels=torch.tensor([1, 2]), masks=gt)]
                masks = mask_logits([[[[1, 0], [0, 0]], [[1, 1], [1, 1]], [[1, 1], [0, 0]],
                                      [[0, 0], [1, 1]], [[0, 0], [1, 1]], [[0, 0], [0, 0]]]]).requires_grad_()
                logits = torch.zeros(1, 6, 4, requires_grad=True)
                aux_masks = torch.full_like(masks, -10., requires_grad=True)
                aux_logits = torch.zeros_like(logits, requires_grad=True)
                outputs = dict(pred_logits=logits, pred_masks=masks,
                               aux_outputs=[dict(pred_logits=aux_logits, pred_masks=aux_masks)])
                matches = [(torch.arange(6), torch.arange(6))]
                counts = [];teacher_indices = []
                original = c.get_loss
                def get_loss(loss, out, targets, indices, count):
                    counts.append(count)
                    return original(loss, out, targets, indices, count)
                def teacher(out, indices, targets):
                    teacher_indices.extend(indices)
                    return {'loss_object_decoder': out['pred_logits'].sum() * 0}
                with patch.object(c.matcher, 'forward', return_value=matches), patch.object(c, 'get_loss', side_effect=get_loss), patch.object(c, 'loss_object_decoder', side_effect=teacher):
                    losses = c(outputs, target)
                self.assertEqual(teacher_indices[0][0].tolist(), [2, 3, 4])
                self.assertEqual(teacher_indices[0][1].tolist(), [0, 1, 1])
                self.assertEqual(counts, [3., 3., 2., 2.])
                sum(losses.values()).backward()
                self.assertLess(logits.grad[0, 0, -1], 0)  # rejected: no-object
                self.assertLess(logits.grad[0, 2, 1], 0)   # retained: GT class
                self.assertEqual(masks.grad[0, [0, 1, 5]].abs().sum(), 0)
                self.assertGreater(masks.grad[0, [2, 3, 4]].abs().sum(), 0)
                self.assertTrue(torch.isfinite(aux_masks.grad).all())
                self.assertEqual(aux_masks.grad[0, [1, 2, 4, 5]].abs().sum(), 0)
                self.assertGreater(aux_masks.grad[0, [0, 3]].abs().sum(), 0)
                self.assertLess(aux_logits.grad[0, 0, 1], 0)  # auxiliary base match
                self.assertLess(aux_logits.grad[0, 2, -1], 0)  # auxiliary rejection


if __name__ == '__main__':
    unittest.main()
