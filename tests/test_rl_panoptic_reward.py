"""Parity with the previous RL reward on identical masks and trajectories."""
from pathlib import Path
import importlib.util
from unittest.mock import patch

import pytest
import torch
from torch import nn

from mask2former.modeling.criterion import SetCriterion

spec = importlib.util.spec_from_file_location(
    "object_reward_reference", Path(__file__).with_name("object_reward_reference.py")
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def criterion(size, paint_order):
    result = SetCriterion.__new__(SetCriterion)
    nn.Module.__init__(result)
    result.num_classes = 3
    result.thing_class_ids = (1, 2)
    result.object_rl_reward_size = size
    result.object_rl_paint_order = paint_order
    return result


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("paint_order", ["forward", "reverse"])
@pytest.mark.parametrize("reward_size", [0, 2, 8])
def test_reward_parity(device, paint_order, reward_size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    generator = torch.Generator().manual_seed(421)
    masks = torch.randn(4, 7, 4, 4, generator=generator, device="cpu").to(device)
    # Perfect objects, two stuff queries to merge, empty query and occlusion.
    masks[0].fill_(-10)
    masks[0, 0, :, :2] = 10
    masks[0, 1, :, 2:] = 10
    masks[0, 2].fill_(10)
    masks[0, 3, :2] = 10
    masks[0, 4, 2:] = 10
    logits = torch.randn(4, 7, 4, generator=generator).to(device)
    logits[0].fill_(-10)
    logits[0, [0, 1], 1] = 10
    logits[0, 2, 2] = 10
    logits[0, [3, 4], 0] = 10
    outputs = {"pred_masks": masks.requires_grad_(), "pred_logits": logits.requires_grad_()}
    targets = []
    for image in range(4):
        gt = torch.tensor([[1, 1, 2, 2], [1, 1, 2, 2],
                           [1, 1, 2, 2], [0, 0, 0, 3]], device=device)
        if image == 1:
            gt = torch.zeros_like(gt)  # Empty target.
        labels = [1, 1, 2] if image != 1 else []
        target_masks = torch.stack([gt == i for i in range(1, 4)]) if labels else gt.new_zeros((0, 4, 4)).bool()
        targets.append({"masks": target_masks, "labels": torch.tensor(labels, device=device, dtype=torch.long)})
    evaluator = criterion(reward_size, paint_order)
    orders = [[0, 1, 2, 3, 4, 5], [0, 1], [], [2, 0, 6, 2, 3]]
    expected = reference.object_reward(evaluator, orders, outputs, targets)
    actual = evaluator.object_reward(orders, outputs, targets)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual.device.type == device
    assert actual.dtype == torch.float32
    assert not actual.requires_grad


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reward_large_batch_with_shared_target_size(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    evaluator = criterion(0, "forward")
    masks = torch.ones(75, 1, 2, 2, device=device)
    logits = torch.tensor([0., 10., 0., -10.], device=device).expand(75, 1, 4)
    outputs = dict(pred_masks=masks, pred_logits=logits)
    targets = [dict(masks=torch.ones(1, size, size, device=device, dtype=torch.bool),
                    labels=torch.tensor([1], device=device)) for size in [4] * 75]
    orders = [[0]] * 75
    expected = reference.object_reward(evaluator, orders, outputs, targets)
    actual = evaluator.object_reward(orders, outputs, targets)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, torch.ones(75, device=device))


def test_terminal_rewards_use_one_batched_api_call():
    evaluator = criterion(0, "forward")
    outputs = dict(pred_masks=torch.ones(3, 1, 2, 2),
                   pred_logits=torch.tensor([0., 10., 0., -10.]).expand(3, 1, 4))
    targets = [dict(masks=torch.ones(1, 2, 2, dtype=torch.bool), labels=torch.tensor([1]))] * 3
    # Load the API through the production path before wrapping it.
    evaluator.object_reward([[0]] * 3, outputs, targets)
    import panoptic_evaluator
    with patch.object(panoptic_evaluator, "panoptic_quality", wraps=panoptic_evaluator.panoptic_quality) as api:
        rewards = evaluator.object_reward([[0]] * 3, outputs, targets)
    assert api.call_count == 1
    assert api.call_args.args[0].segment_map.shape == (3, 2, 2)
    torch.testing.assert_close(rewards, torch.ones(3))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reward_strict_iou_threshold(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    evaluator = criterion(0, "forward")
    masks = torch.tensor([[[[1., 1., -1., -1.]]], [[[1., 1., 1., -1.]]]], device=device)
    outputs = dict(pred_masks=masks,
                   pred_logits=torch.tensor([0., 10., 0., -10.], device=device).expand(2, 1, 4))
    targets = [dict(masks=torch.ones(1, 1, 4, dtype=torch.bool, device=device),
                    labels=torch.tensor([1], device=device))] * 2
    rewards = evaluator.object_reward([[0], [0]], outputs, targets)
    torch.testing.assert_close(rewards, torch.tensor([0., .75], device=device))


def test_terminal_reward_does_not_select_the_best_prefix():
    evaluator = criterion(0, "forward")
    masks = torch.tensor([[[[10., 10., -10., -10.]],
                           [[-10., -10., 10., 10.]],
                           [[10., 10., 10., 10.]]]])
    logits = torch.full((1, 3, 4), -10.)
    logits[0, :2, 1] = 10.
    logits[0, 2, 2] = 10.
    outputs = dict(pred_masks=masks, pred_logits=logits)
    targets = [dict(masks=masks[0, :2] > 0, labels=torch.tensor([1, 1]))]
    torch.testing.assert_close(evaluator.object_reward([[0, 1]], outputs, targets), torch.ones(1))
    torch.testing.assert_close(evaluator.object_reward([[0, 1, 2]], outputs, targets), torch.zeros(1))


@pytest.mark.parametrize("paint_order", ["forward", "reverse"])
@pytest.mark.parametrize("reward_size", [0, 2, 8])
def test_sample_major_rewards_match_individual_rollouts(paint_order, reward_size):
    evaluator = criterion(reward_size, paint_order)
    generator = torch.Generator().manual_seed(91)
    outputs = dict(pred_masks=torch.randn(2, 5, 4, 4, generator=generator),
                   pred_logits=torch.randn(2, 5, 4, generator=generator))
    targets = [dict(masks=torch.rand(3, 4, 4, generator=generator) > .5,
                    labels=torch.tensor([0, 1, 2])),
               dict(masks=torch.zeros(0, 4, 4, dtype=torch.bool),
                    labels=torch.empty(0, dtype=torch.long))]
    sampled = [[0, 1, 0, 3], [], [4, 2], [1, 2, 3]]
    greedy = [[], [0, 4]]
    expected = torch.stack([reference.object_reward(evaluator, orders, outputs, targets)
                            for orders in (sampled[:2], sampled[2:], greedy)])
    actual, baseline = evaluator._object_rollout_rewards(sampled, greedy, outputs, targets)
    torch.testing.assert_close(actual, expected[:2])
    torch.testing.assert_close(baseline, expected[2])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("paint_order", ["forward", "reverse"])
def test_coco_reward_preserves_person_instances_and_merges_each_stuff_class(device, paint_order):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    evaluator = criterion(0, paint_order)
    evaluator.num_classes = 133
    evaluator.thing_class_ids = tuple(range(80))
    # Two person instances and two distinct stuff categories, each represented
    # by two partial proposals. Perfect painting must score PQ=1 in every class.
    regions = torch.zeros(6, 2, 6, device=device, dtype=torch.bool)
    for query in range(6):
        regions[query, :, query] = True
    logits = torch.full((1, 6, 134), -10., device=device)
    labels = torch.tensor([0, 0, 80, 80, 81, 81], device=device)
    logits[0, torch.arange(6, device=device), labels] = 10.
    outputs = dict(pred_masks=regions[None].float()*20-10, pred_logits=logits)
    gt_masks = torch.stack((regions[0], regions[1], regions[2] | regions[3], regions[4] | regions[5]))
    targets = [dict(masks=gt_masks, labels=torch.tensor([0, 0, 80, 81], device=device))]
    actual = evaluator.object_reward([list(range(6))], outputs, targets)
    torch.testing.assert_close(actual, torch.ones(1, device=device))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_coco_reward_matches_inference_painter_for_overlap_and_repeated_queries(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from mask2former.modeling.panoptic_paint import paint_panoptic_masks
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "panoptic-evaluator"))
    from panoptic_evaluator import PanopticBatch, panoptic_quality
    generator = torch.Generator().manual_seed(73)
    masks = torch.randn(1, 7, 8, 8, generator=generator).to(device)
    labels = torch.tensor([0, 0, 80, 80, 81, 81, 2], device=device)
    logits = torch.full((1, 7, 134), -10., device=device)
    logits[0, torch.arange(7, device=device), labels] = 10.
    order = torch.tensor([0, 2, 4, 1, 3, 2, 6], device=device)
    isthing = torch.arange(133, device=device) < 80
    for paint_order in ["forward", "reverse"]:
        evaluator = criterion(0, paint_order)
        evaluator.num_classes = 133
        evaluator.thing_class_ids = tuple(range(80))
        gt_maps, gt_labels = paint_panoptic_masks(masks[:, order], labels[order][None],
            torch.ones(1, len(order), device=device, dtype=torch.bool), isthing,
            reverse=paint_order == "reverse")
        # Prediction segment IDs need not match the GT IDs. This checks actual
        # region/category equivalence to the production inference painter.
        gt_ids = [i for i, label in enumerate(gt_labels[0].tolist()) if i and label >= 0]
        targets = [dict(masks=torch.stack([gt_maps[0] == i for i in gt_ids]),
                        labels=gt_labels[0, gt_ids].long())]
        actual = evaluator.object_reward([order.tolist()], dict(pred_masks=masks, pred_logits=logits), targets)
        expected = panoptic_quality(PanopticBatch(gt_maps, gt_labels),
                                   PanopticBatch(gt_maps, gt_labels), 133).float()
        torch.testing.assert_close(actual, expected)


def test_batch_pq_uses_global_counts_and_counterfactual_credit():
    from mask2former.modeling.criterion import batch_pq_counterfactual_rewards
    # Greedy: class 0 has one TP plus three FN; class 1 has one TP.
    # Fixing the three FN improves global PQ by .3, whereas image PQ
    # improves by .5. Batch-size scaling gives a reward difference of .6.
    stats = torch.tensor([
        [[1., 1., 0., 0.], [0., 0., 0., 0.]],
        [[3., 3., 0., 0.], [1., 1., 0., 0.]],
        [[1., 1., 0., 0.], [0., 0., 0., 0.]],
        [[0., 0., 0., 3.], [1., 1., 0., 0.]],
    ])
    sampled, baseline = batch_pq_counterfactual_rewards(stats, 2)
    torch.testing.assert_close(sampled, torch.tensor([[0., .6]]))
    torch.testing.assert_close(baseline, torch.zeros(2))
    # Distributed greedy context includes an identical second rank.
    with patch('mask2former.modeling.criterion.is_dist_avail_and_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=2), \
         patch('torch.distributed.all_reduce', side_effect=lambda x: x.mul_(2)):
        sampled, _ = batch_pq_counterfactual_rewards(stats, 2)
    # Global class 0: (2+3)/(5+.5*3) versus 2/5.
    torch.testing.assert_close(sampled, torch.tensor([[0., (5/6.5-.4)*2]]))


def test_reward_statistics_reproduce_image_pq():
    evaluator = criterion(0, 'forward')
    outputs = dict(pred_masks=torch.ones(2, 1, 2, 2),
                   pred_logits=torch.tensor([0., 10., 0., -10.]).expand(2, 1, 4))
    targets = [dict(masks=torch.ones(1, 2, 2, dtype=torch.bool), labels=torch.tensor([1]))] * 2
    stats = evaluator.object_reward([[0], []], outputs, targets, return_stats=True)
    assert stats.shape == (2, 3, 4)
    torch.testing.assert_close(stats[:, 1], torch.tensor([[1., 1., 0., 0.], [0., 0., 0., 1.]]))
    evaluator.object_rl_reward = 'batch_pq'
    sampled, baseline = evaluator._object_rollout_rewards([[0], []], [[0], [0]], outputs, targets)
    torch.testing.assert_close(sampled, torch.tensor([[0., -2/3]]))
    torch.testing.assert_close(baseline, torch.zeros(2))


def test_batch_pq_logs_absolute_scores_separately_from_signed_rewards():
    from detectron2.utils.events import EventStorage
    from mask2former.utils.rl_logging import record_batch_pq_diagnostics, record_rl_diagnostics
    stats = torch.tensor([[[0., 0., 0., 1.]], [[1., 1., 0., 0.]]])
    with EventStorage() as storage:
        record_batch_pq_diagnostics(stats, 1)
        record_rl_diagnostics(torch.tensor([[-1.]]), torch.zeros(1),
                              torch.tensor([[-1.]]), torch.zeros(1, 1), torch.tensor(0.),
                              reward_mode='batch_pq')
        metrics = storage.latest()
        assert metrics['rl/pq/greedy_batch_pq'][0] == 100
        assert metrics['rl/pq/sample_batch_pq'][0] == 0
        assert metrics['rl/reward/sample_scaled_delta_pq'][0] == -100
        assert metrics['rl/reward/greedy_scaled_delta_pq'][0] == 0
        assert 'rl/reward/greedy_pq' not in metrics
