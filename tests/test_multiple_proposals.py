"""Repeated matching, alternative teachers, and standard class supervision."""
import unittest
from unittest.mock import patch
import torch
from detectron2.utils.events import EventStorage
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from mask2former.modeling.multiple_proposals import repeat_object_targets, generate_proposal_permutations
from mask2former.modeling.teacher_permutations import generate_permutations, select_best_permutations


def criterion(mask_type='point', copies=2):
    matcher = HungarianMatcher(cost_class=2, cost_mask=5, cost_dice=5,
                               cost_query_bias=0, num_points=32, mask_loss_type=mask_type)
    matcher.standard_classification = True
    c = SetCriterion(3, matcher, {}, .1, ['labels', 'masks'], 32, 3., .75,
                     mask_loss_type=mask_type, fused_thing_masks=False)
    c.standard_classification = True
    c.proposals_per_gt = copies
    c.object_pretrain_best_of_k = 4
    return c


class MultipleProposalsTests(unittest.TestCase):
    def test_repeated_targets_exclude_parts_and_preserve_original(self):
        t = dict(labels=torch.tensor([1, 2]), masks=torch.eye(2).reshape(2, 1, 2), part_masks=torch.ones(9, 1, 2))
        result = repeat_object_targets([t], 2)[0]
        self.assertEqual(set(result), {'labels', 'masks'})
        self.assertEqual(result['labels'].tolist(), [1, 1, 2, 2])
        self.assertEqual(t['labels'].tolist(), [1, 2])
        torch.testing.assert_close(result['masks'][::2], t['masks'])

    def test_teachers_pick_exactly_one_alternative_per_object(self):
        torch.manual_seed(42)
        indices = [(torch.tensor([1, 7, 3, 9]), torch.tensor([0, 0, 2, 2])),
                   (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))]
        candidates = generate_proposal_permutations(indices, 64, 'cpu')
        seen = set()
        for order, empty in zip(candidates[::2], candidates[1::2]):
            self.assertEqual(len(order), 2)
            self.assertEqual(len(set(order) & {1, 7}), 1)
            self.assertEqual(len(set(order) & {3, 9}), 1)
            self.assertEqual(empty, [])
            seen.update(order)
        self.assertEqual(seen, {1, 7, 3, 9})

    def test_best_teacher_selection_is_sample_major(self):
        self.assertEqual(select_best_permutations([[0], [1], [2], [3]], torch.tensor([[.1, .9], [.8, .2]])), [[2], [1]])
        with self.assertRaises(ValueError):
            select_best_permutations([[0]], torch.tensor([[float('nan')]]))
        self.assertEqual(generate_permutations([(torch.tensor([2]), torch.tensor([0]))], 2, 'cpu'), [[2], [2]])

    def test_standard_classification_trains_unmatched_no_object(self):
        c = criterion(); logits = torch.zeros(1, 3, 4, requires_grad=True)
        loss = c.loss_labels({'pred_logits': logits}, [{'labels': torch.tensor([1])}], [(torch.tensor([0]), torch.tensor([0]))], 1)['loss_ce']
        loss.backward()
        self.assertLess(logits.grad[0, 0, 1], 0)
        self.assertLess(logits.grad[0, 1, 3], 0)
        self.assertGreater(logits.grad[0, 1, 1], 0)

    def test_standard_matching_includes_no_object_probability(self):
        c = criterion()
        c.matcher.cost_mask = c.matcher.cost_dice = 0
        outputs = {'pred_logits': torch.tensor([[[0., 5., 0., 20.], [0., 2., 0., -20.]]]), 'pred_masks': torch.zeros(1, 2, 2, 2)}
        target = [{'labels': torch.tensor([1]), 'masks': torch.ones(1, 2, 2)}]
        self.assertEqual(c.matcher(outputs, target)[0][0].tolist(), [1])
        c.matcher.standard_classification = False
        self.assertEqual(c.matcher(outputs, target)[0][0].tolist(), [0])

    def test_point_and_block_losses_auxiliary_overflow_and_unrepeated_reward(self):
        for kind in ['point', 'block']:
            with self.subTest(kind=kind), EventStorage():
                c = criterion(kind)
                target = [dict(labels=torch.tensor([1, 2, 1]), masks=(torch.rand(3, 8, 8) > .5))]
                out = dict(pred_logits=torch.randn(1, 4, 4, requires_grad=True), pred_masks=torch.randn(1, 4, 2, 2, requires_grad=True))
                out['aux_outputs'] = [dict(pred_logits=torch.randn(1, 4, 4, requires_grad=True), pred_masks=torch.randn(1, 4, 2, 2, requires_grad=True))]
                captured = []
                def teacher(outputs, indices, targets):
                    captured.append(indices)
                    self.assertIs(targets, target)
                    return {'loss_object_decoder': outputs['pred_logits'].sum() * 0}
                with patch.object(c, 'loss_object_decoder', side_effect=teacher):
                    losses = c(out, target)
                self.assertEqual(len(captured[0][0][0]), 4)
                self.assertTrue((captured[0][0][1] < 3).all())
                self.assertEqual(set(losses), {'loss_ce', 'loss_mask', 'loss_dice', 'loss_ce_0', 'loss_mask_0', 'loss_dice_0', 'loss_object_decoder'})
                sum(losses.values()).backward()
                self.assertTrue(torch.isfinite(out['pred_masks'].grad).all())
                self.assertGreater(out['pred_masks'].grad.abs().sum(), 0)
                self.assertGreater(out['aux_outputs'][0]['pred_masks'].grad.abs().sum(), 0)


if __name__ == '__main__':
    unittest.main()
