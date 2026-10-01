import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location(
    "synthetic_panoptic_evaluator",
    Path(__file__).resolve().parents[1]
    / "mask2former/evaluation/synthetic_scene_panoptic_evaluation.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Evaluator = module.SyntheticScenePanopticEvaluator


class OracleThresholdTest(unittest.TestCase):
    def check_prediction(self, gt, info, proposals):
        prediction, segments = Evaluator._oracle_assignment(gt, info, proposals)
        evaluator = Evaluator(num_classes=2)
        evaluator.reset()
        evaluator._accumulate_image(gt, info, prediction, segments)
        self.assertEqual(evaluator._stats[:, 2].sum().item(), 0)
        self.assertEqual(evaluator._stats[:, 1].sum().item(), len(segments))
        return prediction, segments

    def test_strict_threshold(self):
        gt = torch.ones((1, 4), dtype=torch.long)
        info = [{"id": 1, "category_id": 1}]
        for logits, expected in (([1., -1., -1., -1.], 0),
                                 ([1., 1., -1., -1.], 0),
                                 ([1., 1., 1., -1.], 1)):
            _, segments = self.check_prediction(gt, info, torch.tensor([[logits]]))
            self.assertEqual(len(segments), expected)

    def test_overlap_pruning(self):
        gt = torch.tensor([[1, 1, 1, 2, 2, 2]])
        info = [{"id": 1, "category_id": 0}, {"id": 2, "category_id": 1}]
        # Both original masks have IoU 0.6, but the second steals two pixels
        # from the first, making its painted IoU fall below the threshold.
        proposals = torch.tensor([[[1., 1., 1., 1., 1., -1.]],
                                  [[2., 2., -1., 2., 2., 2.]]])
        _, segments = self.check_prediction(gt, info, proposals)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["category_id"], 1)

    def test_all_queries_duplicates_and_void(self):
        gt = torch.tensor([[1, 1, 2, 2, 0]])
        info = [{"id": 1, "category_id": 0}, {"id": 2, "category_id": 1}]
        proposals = torch.tensor([[[1., 1., 1., 1., 1.]],
                                  [[2., 2., -1., -1., 2.]],
                                  [[-1., -1., 2., 2., -1.]],
                                  [[1., 1., -1., -1., -1.]]])
        _, segments = self.check_prediction(gt, info, proposals)
        self.assertEqual(len(segments), 2)

    def test_empty_proposals(self):
        gt = torch.tensor([[1, 1]])
        _, segments = self.check_prediction(
            gt, [{"id": 1, "category_id": 0}], torch.empty((0, 1, 2)))
        self.assertEqual(segments, [])


if __name__ == "__main__":
    unittest.main()
