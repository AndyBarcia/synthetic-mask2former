"""Distributed RL diagnostics; rewards are reported in PQ points (0–100)."""

import logging

import torch
import torch.distributed as dist
from detectron2.utils.events import EventWriter, get_event_storage


@torch.no_grad()
def record_rl_diagnostics(sampled_reward, greedy_reward, sampled, greedy,
                          sampled_stats, greedy_stats, log_probability):
    # Keep criterion calls outside a training EventStorage usable (e.g. diagnostics).
    try:
        storage = get_event_storage()
    except AssertionError:
        return
    advantage = sampled_reward - greedy_reward
    values = {
        "images": sampled_reward.new_tensor(sampled_reward.numel()),
        "reward/sample_pq": sampled_reward.sum() * 100,
        "reward/greedy_pq": greedy_reward.sum() * 100,
        "advantage/mean_pq": advantage.sum() * 100,
        "advantage/second_moment": advantage.square().sum() * 10000,
        "advantage/abs_pq": advantage.abs().sum() * 100,
        "advantage/win_fraction": (advantage > 1e-6).float().sum(),
        "advantage/tie_fraction": (advantage.abs() <= 1e-6).float().sum(),
        "advantage/loss_fraction": (advantage < -1e-6).float().sum(),
        "sequence/identical_fraction": sampled_reward.new_tensor(sum(a == b for a, b in zip(sampled, greedy))),
        "policy/sample_nll": -log_probability.detach().sum(),
    }
    for name, stats in (("sample", sampled_stats), ("greedy", greedy_stats)):
        for key, value in stats.items():
            values[f"{name}/{key}"] = value.sum()
    keys = list(values)
    totals = torch.stack([values[key].float() for key in keys])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    totals = dict(zip(keys, totals.cpu().tolist()))
    count = totals.pop("images")
    metrics = {}
    for key in list(totals):
        if key.startswith(("sample/", "greedy/")):
            name, stat = key.split("/")
            denominator = count if stat in ("length", "truncated", "actions") else max(totals[f"{name}/actions"], 1)
        elif key == "policy/sample_nll":
            denominator = max(totals["sample/actions"], 1)
        else:
            denominator = count
        metrics[key] = totals[key] / denominator
    second_moment = metrics.pop("advantage/second_moment")
    metrics["advantage/std_pq"] = max(0, second_moment - metrics["advantage/mean_pq"] ** 2) ** 0.5
    for key, value in metrics.items():
        storage.put_scalar(f"rl/{key}", value)


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
        for group in ("reward", "advantage", "sequence", "policy", "sample", "greedy"):
            fields = [f"{name.split('/')[-1]}={storage.history(name).median(self.window_size):.4g}"
                      for name in names if name.startswith(f"rl/{group}/")]
            if fields:
                logger.info("RL iter: %d %s: %s", storage.iter, group, "  ".join(fields))
        self.last_iteration = max(latest[name][1] for name in names)
