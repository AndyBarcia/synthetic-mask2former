import unittest
import torch
from mask2former.modeling.yolox.matcher import YOLOXMatcher
from mask2former.modeling.matcher import HungarianMatcher


class TeacherMatchingTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.proposal = dict(masks=torch.full((3, 4, 4), -10.),
                             labels=torch.zeros(3, dtype=torch.long),
                             class_logits=torch.tensor([[8., -8., -8.],
                                                        [7., -7., -7.],
                                                        [6., -6., -6.]]))
        masks = torch.zeros(2, 16, 16)
        masks[0, :8] = 1
        masks[1, 8:] = 1
        self.target = dict(labels=torch.tensor([1, 2]), masks=masks)

    def test_wrong_class_and_zero_iou_still_cover_every_gt(self):
        matcher = YOLOXMatcher(3, {1, 2}, object_iou_threshold=1.)
        matched = matcher.match_objects([self.proposal], [self.target])[0]
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(matched.unique()), 2)

    def test_uses_full_class_distribution_instead_of_argmax_labels(self):
        matcher = YOLOXMatcher(3, {1, 2}, cost_class=1., cost_mask=0., cost_dice=0.)
        self.proposal['class_logits'] = torch.tensor([[8., -8., -8.],
                                                      [0., 8., -8.],
                                                      [0., -8., 8.]])
        matched = matcher.match_objects([self.proposal], [self.target])[0]
        self.assertEqual(set(matched.tolist()), {1, 2})

    def test_mask_cost_selects_geometry_when_classes_are_equal(self):
        matcher = YOLOXMatcher(3, {1, 2}, cost_class=0., cost_mask=5., cost_dice=5.)
        self.proposal['masks'][1, :2] = 10.
        self.proposal['masks'][2, 2:] = 10.
        matched = matcher.match_objects([self.proposal], [self.target])[0]
        self.assertEqual(set(matched.tolist()), {1, 2})

    def test_matches_baseline_hungarian_with_same_random_points(self):
        matcher = YOLOXMatcher(3, {1, 2}, num_points=128)
        baseline = HungarianMatcher(cost_class=2., cost_mask=5., cost_dice=5., num_points=128)
        logits = torch.cat((self.proposal['class_logits'], torch.zeros(3, 1)), -1)
        torch.manual_seed(9)
        expected = baseline(dict(pred_logits=logits[None], pred_masks=self.proposal['masks'][None]),
                            [self.target])[0][0]
        torch.manual_seed(9)
        actual = matcher.match_objects([self.proposal], [self.target])[0]
        torch.testing.assert_close(actual, expected)

    def test_empty_scene_and_fewer_proposals_than_objects(self):
        matcher = YOLOXMatcher(3, {1, 2})
        empty = dict(labels=self.target['labels'][:0], masks=self.target['masks'][:0])
        self.assertEqual(len(matcher.match_objects([self.proposal], [empty])[0]), 0)
        proposal = {k: v[:1] for k, v in self.proposal.items()}
        self.assertEqual(len(matcher.match_objects([proposal], [self.target])[0]), 1)


if __name__ == '__main__':
    unittest.main()
