import math
import unittest
from unittest.mock import patch
import torch
from mask2former.modeling.yolox.criterion import YOLOXCriterion
from mask2former.modeling.yolox.matcher import YOLOXMatcher
from mask2former.modeling.yolox.object_decoder import ProposalObjectDecoder
from tests.test_native_yolox import model, batch


class GroupedTeacherTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_group_probability_sum_and_gradient_direction(self):
        logits = torch.tensor([[[math.log(.2), math.log(.3), math.log(.5)]]], requires_grad=True)
        weights = torch.tensor([[[1., 0., 0.]]])
        loss = YOLOXCriterion.grouped_sequence_loss(logits, weights, [{0: [0, 1]}])
        self.assertAlmostEqual(float(loss), -math.log(.5), places=6)
        loss.backward()
        self.assertTrue(bool((logits.grad[0, 0, :2] < 0).all()))
        self.assertGreater(float(logits.grad[0, 0, 2]), 0)

    def test_trie_edge_counts_and_eof_match_expanded_paths(self):
        logits = torch.randn(1, 3, 4, requires_grad=True)
        weights = torch.tensor([[[2., 0., 1., 0.], [0., 0., 0., 2.], [0., 0., 0., 0.]]])
        logp = logits.log_softmax(-1)
        expected = -(2 * torch.logsumexp(logp[0, 0, :2], 0) + logp[0, 0, 2] + 2 * logp[0, 1, 3]) / 5
        actual = YOLOXCriterion.grouped_sequence_loss(logits, weights, [{0: [0, 1], 2: [2]}])
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertEqual(float(logits.grad[0, 2].abs().sum()), 0.)

    def test_group_maps_exact_mask_positives_through_score_sort(self):
        m = model().criterion
        outputs = dict(proposals=[dict(dense_ids=torch.tensor([9, 4, 7, -1]),
                                       labels=torch.tensor([2, 0, 2, 1]))])
        # In this fixture class 0 is thing, class 1 is stuff. Wrong proposal
        # predictions must not remove mask-positive group members.
        prepared = [dict(labels=torch.tensor([0, 1]))]
        indices = [(torch.tensor([7, 9, 4]), torch.tensor([0, 0, 0]))]
        representatives, groups = m.dynamic_teacher_groups(outputs, prepared, indices)
        self.assertEqual(representatives[0].tolist(), [2, 3])
        self.assertEqual(groups[0], {2: [2, 0, 1], 3: [3]})
        outputs['proposals'][0]['dense_ids'][0] = 99
        with self.assertRaisesRegex(ValueError, 'increase TOPK'):
            m.dynamic_teacher_groups(outputs, prepared, indices)

    def test_conflict_repair_covers_gt_and_preserves_disjoint_groups(self):
        matcher = YOLOXMatcher(3, {1, 2}, ensure_gt_coverage=True)
        match = torch.tensor([[True, True, False], [False, False, False]])
        cost = torch.tensor([[3., 1., 5.], [4., 3., 2.]])
        ids, gt = matcher._finish_assignment(match, cost, torch.tensor([10, 11, 12]))
        self.assertEqual(len(ids.unique()), len(ids))
        self.assertEqual(set(gt.tolist()), {0, 1})
        self.assertEqual(ids[gt == 0][0].item(), 11)
        # With no free candidates, steal only from a GT with multiple matches.
        ids, gt = matcher._finish_assignment(match[:, :2], cost[:, :2], torch.tensor([10, 11]))
        self.assertEqual(set(gt.tolist()), {0, 1})

    def grouped_model(self):
        m = model().train()
        m.sem_seg_head.num_queries = m.num_queries = 86  # all 84 things + 2 stuff
        m.criterion.object_decoder = ProposalObjectDecoder(256, 32, 86, 1, 4, 64, prefix_tree_branches=2)
        m.criterion.object_dec_supervision = 'dynamic_k_grouped'
        m.criterion.matcher.ensure_gt_coverage = True
        return m

    def test_training_reuses_mask_assignment_without_hungarian_and_backprops(self):
        m = self.grouped_model()
        # The token adapter starts with a zero final projection. Use a nonzero
        # projection to test its upstream dynamic-parameter gradient path.
        torch.nn.init.normal_(m.sem_seg_head.token_adapter[-1].weight, std=.01)
        matcher = m.criterion.matcher
        with patch.object(matcher, 'forward', wraps=matcher.forward) as assignment, \
             patch.object(matcher, 'match_objects', side_effect=AssertionError('Hungarian used')):
            losses = m(batch())
        self.assertEqual(assignment.call_count, 1)
        losses['loss_object_decoder'].backward()
        for module in [m.sem_seg_head.predictor.kernel, m.sem_seg_head.predictor.dynamic,
                       m.sem_seg_head.token_adapter, m.criterion.object_decoder]:
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)
        m.zero_grad(set_to_none=True)
        losses = m(batch(empty=True))
        sum(losses.values()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))


if __name__ == '__main__':
    unittest.main()
