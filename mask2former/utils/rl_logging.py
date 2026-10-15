"""Distributed RL diagnostics; absolute PQ and signed reward changes are distinct."""

import logging

import torch
import torch.distributed as dist
from detectron2.utils.events import EventWriter, get_event_storage


@torch.no_grad()
def record_rl_diagnostics(sampled_reward, greedy_reward, advantage,
                          log_probability, rl_loss, sampled_lengths=None,
                          greedy_lengths=None, reward_mode="image_pq"):
    # Keep criterion calls outside a training EventStorage usable (e.g. diagnostics).
    try:
        storage = get_event_storage()
    except AssertionError:
        return
    k, batch_size = sampled_reward.shape
    values = {
        "images": sampled_reward.new_tensor(batch_size),
        "samples": sampled_reward.new_tensor(k * batch_size),
        "reward/sample_pq": sampled_reward.sum() * 100,
        "reward/best_sample_pq": sampled_reward.max(0).values.sum() * 100,
        "reward/greedy_pq": greedy_reward.sum() * 100,
        "advantage/mean_pq": advantage.sum() * 100,
        "advantage/second_moment": advantage.square().sum() * 10000,
        "advantage/abs_pq": advantage.abs().sum() * 100,
        "advantage/win_fraction": (advantage > 1e-6).float().sum(),
        "advantage/tie_fraction": (advantage.abs() <= 1e-6).float().sum(),
        "advantage/loss_fraction": (advantage < -1e-6).float().sum(),
        "policy/sample_nll": -log_probability.detach().sum(),
        "loss": rl_loss.detach() * batch_size,
    }
    if sampled_lengths is not None:
        values["sequence/sample_masks"] = sampled_lengths.sum()
        values["sequence/greedy_masks"] = greedy_lengths.sum()
    keys = list(values)
    totals = torch.stack([values[key].float() for key in keys])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    totals = dict(zip(keys, totals.cpu().tolist()))
    count = totals.pop("images")
    sample_count = totals.pop("samples")
    metrics = {}
    for key in list(totals):
        denominator = (sample_count if key in (
            "reward/sample_pq", "advantage/mean_pq", "advantage/second_moment",
            "advantage/abs_pq", "advantage/win_fraction", "advantage/tie_fraction",
            "advantage/loss_fraction", "policy/sample_nll", "sequence/sample_masks",
        ) else count)
        metrics[key] = totals[key] / denominator
    second_moment = metrics.pop("advantage/second_moment")
    metrics["advantage/std_pq"] = max(0, second_moment - metrics["advantage/mean_pq"] ** 2) ** 0.5
    for key, value in metrics.items():
        if reward_mode in ("batch_pq", "batch_rq") and key.startswith("reward/"):
            key = key.replace("_pq", "_scaled_delta_pq")
        if reward_mode == "batch_rq":
            key = key.replace("_pq", "_rq")
        storage.put_scalar(f"rl/{key}", value)


@torch.no_grad()
def record_best_of_n_diagnostics(sampled_reward, greedy_reward, best_reward,
                                 winners, greedy_orders, best_trajectories,
                                 loss, token_count, metric_group="best_of_n",
                                 mask_nll=None, eof_nll=None, mask_count=None,
                                 reward_mode="image_pq"):
    try:
        storage = get_event_storage()
    except AssertionError:
        return
    k, batch_size = sampled_reward.shape
    values = {
        "images": sampled_reward.new_tensor(batch_size),
        "samples": sampled_reward.new_tensor(k * batch_size),
        "reward/sample_pq": sampled_reward.sum() * 100,
        "reward/greedy_pq": greedy_reward.sum() * 100,
        "reward/best_sample_pq": best_reward.sum() * 100,
        f"{metric_group}/win_fraction": winners.float().sum(),
        f"{metric_group}/mean_positive_gain_pq": (
            best_reward - greedy_reward
        ).clamp_min(0).sum() * 100,
        "sequence/greedy_masks": sampled_reward.new_tensor(sum(map(len, greedy_orders))),
        "sequence/best_masks": sampled_reward.new_tensor(
            sum(len(trajectory) - 1 for trajectory in best_trajectories)
        ),
        "tokens": sampled_reward.new_tensor(token_count),
        f"{metric_group}/{'ce_loss' if metric_group == 'best_of_n' else 'set_loss'}": (
            loss.detach() * token_count
        ),
        "loss": loss.detach() * batch_size,
    }
    if mask_nll is not None:
        values["mask_tokens"] = sampled_reward.new_tensor(mask_count)
        values["set_imitation/mask_nll"] = mask_nll
        values["set_imitation/eof_nll"] = eof_nll
    keys = list(values)
    totals = torch.stack([values[key].float() for key in keys])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    totals = dict(zip(keys, totals.cpu().tolist()))
    count = totals.pop("images")
    sample_count = totals.pop("samples")
    scored_tokens = totals.pop("tokens")
    mask_tokens = totals.pop("mask_tokens", 0)
    for key, value in totals.items():
        if key == "reward/sample_pq":
            denominator = sample_count
        elif key in ("best_of_n/ce_loss", "set_imitation/set_loss"):
            denominator = max(scored_tokens, 1)
        elif key == "set_imitation/mask_nll":
            denominator = max(mask_tokens, 1)
        else:
            denominator = count
        if reward_mode in ("batch_pq", "batch_rq") and key.startswith("reward/"):
            key = key.replace("_pq", "_scaled_delta_pq")
        if reward_mode == "batch_rq":
            key = key.replace("_pq", "_rq")
        storage.put_scalar(f"rl/{key}", value / denominator)


@torch.no_grad()
def record_batch_pq_diagnostics(stats, batch_size):
    """Absolute PQ for greedy and complete sampled distributed batches."""
    try:
        storage = get_event_storage()
    except AssertionError:
        return
    totals = stats.reshape(-1, batch_size, stats.shape[-2], 4).double().sum(1)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    iou, tp, fp, fn = totals.unbind(-1)
    denominator = tp + .5 * (fp + fn)
    pq = (iou / denominator.clamp_min(1)).sum(-1) / (denominator > 0).sum(-1).clamp_min(1)
    storage.put_scalar("rl/pq/greedy_batch_pq", pq[-1].item() * 100)
    storage.put_scalar("rl/pq/sample_batch_pq", pq[:-1].mean().item() * 100)
    rq = (tp / denominator.clamp_min(1)).sum(-1) / (denominator > 0).sum(-1).clamp_min(1)
    storage.put_scalar("rl/rq/greedy_batch_rq", rq[-1].item() * 100)
    storage.put_scalar("rl/rq/sample_batch_rq", rq[:-1].mean().item() * 100)


class RLMetricPrinter(EventWriter):
    """Print RL scalars that Detectron2's standard loss printer omits."""

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.last_iteration = -1

    def write(self):
        storage = get_event_storage()
        latest = storage.latest()
        names = sorted(name for name, (_, iteration) in latest.items()
                       if name.startswith("rl/") and iteration > self.last_iteration)
        if not names:
            return
        logger = logging.getLogger("mask2former.rl")
        for group in ("pq", "rq", "reward", "advantage", "policy", "sequence", "best_of_n",
                      "set_imitation"):
            fields = [f"{name.split('/')[-1]}={storage.history(name).median(self.window_size):.4g}"
                      for name in names if name.startswith(f"rl/{group}/")]
            if fields:
                logger.info("RL iter: %d %s: %s", storage.iter, group, "  ".join(fields))
        if "rl/loss" in names:
            logger.info("RL iter: %d loss: %.4g", storage.iter,
                        storage.history("rl/loss").median(self.window_size))
        self.last_iteration = max(latest[name][1] for name in names)
