from unittest.mock import patch

import torch
from detectron2.utils.events import EventStorage
from mask2former.modeling.criterion import batch_pq_counterfactual_rewards
from mask2former.utils.rl_logging import record_batch_pq_diagnostics, record_rl_diagnostics


def test_rq_ignores_matched_iou_but_rewards_recognition():
    # Identical matches at different IoUs: RQ unchanged, PQ improves.
    stats = torch.tensor([[[.9, 1., 0., 0.]], [[.6, 1., 0., 0.]]])
    rq, baseline = batch_pq_counterfactual_rewards(stats, 1, metric='rq')
    pq, _ = batch_pq_counterfactual_rewards(stats, 1)
    torch.testing.assert_close(rq, torch.zeros(1, 1))
    torch.testing.assert_close(baseline, torch.zeros(1))
    torch.testing.assert_close(pq, torch.tensor([[.3]]))
    # Matching a missing object raises RQ from 0 to 1.
    stats[-1] = torch.tensor([[0., 0., 0., 1.]])
    rq, _ = batch_pq_counterfactual_rewards(stats, 1, metric='rq')
    torch.testing.assert_close(rq, torch.ones(1, 1))


def test_rq_distributed_counterfactual_counts():
    stats = torch.tensor([
        [[.7, 1., 0., 0.], [0., 0., 0., 0.]],
        [[1.8, 3., 0., 0.], [.8, 1., 0., 0.]],
        [[.7, 1., 0., 0.], [0., 0., 0., 0.]],
        [[0., 0., 0., 3.], [.8, 1., 0., 0.]],
    ])
    rq, _ = batch_pq_counterfactual_rewards(stats, 2, metric='rq')
    torch.testing.assert_close(rq, torch.tensor([[0., .6]]))
    with patch('mask2former.modeling.criterion.is_dist_avail_and_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=2), \
         patch('torch.distributed.all_reduce', side_effect=lambda x: x.mul_(2)):
        rq, _ = batch_pq_counterfactual_rewards(stats, 2, metric='rq')
    torch.testing.assert_close(rq, torch.tensor([[0., (5/6.5-.4)*2]]))


def test_rq_logging_uses_correct_metric_names():
    stats = torch.tensor([[[0., 0., 0., 1.]], [[.6, 1., 0., 0.]]])
    with EventStorage() as storage:
        record_batch_pq_diagnostics(stats, 1)
        record_rl_diagnostics(torch.tensor([[-1.]]), torch.zeros(1),
                              torch.tensor([[-1.]]), torch.zeros(1, 1), torch.tensor(0.),
                              reward_mode='batch_rq')
        metrics = storage.latest()
        assert metrics['rl/rq/greedy_batch_rq'][0] == 100
        assert metrics['rl/reward/sample_scaled_delta_rq'][0] == -100
        assert metrics['rl/advantage/mean_rq'][0] == -100
        assert 'rl/reward/sample_scaled_delta_pq' not in metrics
