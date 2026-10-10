"""Combined adapter metrics and decoder confidence regression tests."""
import unittest
from types import SimpleNamespace
import torch
from detectron2.structures import Instances
from mask2former.maskformer_model import MaskFormer
from mask2former.evaluation.synthetic_scene_panoptic_evaluation import SyntheticScenePanopticEvaluator


class CombinedEvaluationTest(unittest.TestCase):
    def test_decoder_scores_and_selection(self):
        model = SimpleNamespace(panoptic_on=True, metadata=SimpleNamespace(
            thing_dataset_id_to_contiguous_id={10: 1}))
        result = MaskFormer.instance_inference(
            model, torch.tensor([[2., 0., -1.], [0., 2., -1.], [0., 3., -1.]]),
            torch.ones(3, 1, 3), torch.tensor([.99, .01, .9]),
            torch.tensor([2, 1]), torch.tensor([.7, .95]))
        torch.testing.assert_close(result.scores, torch.tensor([.7, .95]))
        self.assertEqual(len(result), 2)

    def test_distributed_ap_merges_scores_before_compute(self):
        from panoptic_evaluator import InstanceBatch, InstanceEvaluator
        from mask2former.evaluation.combined_evaluation import instance_state, summarize_instances
        target = InstanceBatch([torch.tensor([[[True, False]]])], [torch.tensor([1])])
        states = []
        whole = InstanceEvaluator(2, device="cpu")
        for mask, score in (([True, False], .5), ([False, True], .9)):
            prediction = InstanceBatch([torch.tensor([[mask]])], [torch.tensor([1])],
                                       scores=[torch.tensor([score])])
            worker = InstanceEvaluator(2, device="cpu")
            worker.update(prediction, target)
            whole.update(prediction, target)
            states.append(instance_state(worker))
        merged = summarize_instances(states, 2)
        self.assertAlmostEqual(merged["AP"], 100 * whole.compute()["ap"].item())
        self.assertLess(merged["AP"], 30)

    def test_combined_perfect_and_empty_masks(self):
        evaluator = SyntheticScenePanopticEvaluator(
            num_classes=2, device="cpu", evaluate_oracle=False, evaluate_instance=True)
        evaluator.reset()
        gt = torch.tensor([[1, 1, 2, 2]], dtype=torch.int32)
        info = [{"id": 1, "category_id": 0}, {"id": 2, "category_id": 1}]
        instances = Instances((1, 4))
        instances.pred_masks = torch.stack([gt == 2, torch.zeros_like(gt, dtype=torch.bool)])
        instances.pred_classes = torch.tensor([1, 1])
        instances.scores = torch.tensor([.9, .99])
        evaluator.process([{"panoptic_ground_truth": (gt, info)}],
                          [{"panoptic_seg": (gt, info), "instances": instances}])
        result = evaluator.evaluate()
        self.assertAlmostEqual(result["panoptic_seg"]["PQ"], 100)
        self.assertAlmostEqual(result["sem_seg"]["mIoU"], 100)
        self.assertAlmostEqual(result["segm"]["AP"], 100)


if __name__ == "__main__":
    unittest.main()
