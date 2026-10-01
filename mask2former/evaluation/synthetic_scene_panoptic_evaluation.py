"""In-memory panoptic evaluation for the online synthetic-scene dataset."""

from collections import OrderedDict
import logging

import torch
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm


class SyntheticScenePanopticEvaluator(DatasetEvaluator):
    """Compute standard PQ/SQ/RQ without serializing predictions or ground truth."""

    def __init__(self, num_classes=6):
        self.num_classes = num_classes

    def reset(self):
        # Columns are IoU sum, true positives, false positives, false negatives.
        self._stats = torch.zeros((self.num_classes, 4), dtype=torch.float64)
        self._oracle_stats = torch.zeros_like(self._stats)

    def process(self, inputs, outputs):
        for input_record, output_record in zip(inputs, outputs):
            gt_map, gt_info = input_record["panoptic_ground_truth"]
            pred_map, pred_info = output_record["panoptic_seg"]
            self._accumulate_image(
                gt_map.detach().to(device="cpu", dtype=torch.int64),
                gt_info,
                pred_map.detach().to(device="cpu", dtype=torch.int64),
                pred_info,
            )
            oracle_map, oracle_info = self._oracle_assignment(
                gt_map, gt_info, output_record["panoptic_proposals"]
            )
            self._accumulate_image(
                gt_map.detach().to(device="cpu", dtype=torch.int64),
                gt_info,
                oracle_map.cpu(),
                oracle_info,
                stats=self._oracle_stats,
            )

    @staticmethod
    def _oracle_assignment(gt_map, gt_info, proposals):
        """Prioritize the number of one-to-one matches with binary IoU > 0.5.

        Classes and proposal selection are oracle decisions. Selected masks keep
        their predicted geometry; overlapping pixels go to the highest logit,
        and pixels outside every selected binary mask remain void. Prune masks
        that fail the same strict threshold after overlap resolution, repainting
        after each removal. This is a heuristic, not a global PQ optimization.
        """
        proposals = proposals.detach()
        gt_map = gt_map.to(device=proposals.device, dtype=torch.int64)
        if proposals.ndim != 3 or proposals.shape[-2:] != gt_map.shape:
            raise ValueError("Panoptic proposals must have shape [Q, H, W] matching ground truth")
        panoptic_map = torch.zeros_like(gt_map)
        if not gt_info or proposals.shape[0] == 0:
            return panoptic_map, []

        gt_masks = torch.stack([gt_map == segment["id"] for segment in gt_info])
        # Ignore void pixels in IoU, as in the standard PQ calculation.
        proposal_masks = (proposals > 0) & (gt_map != 0)[None]
        gt_flat = gt_masks.flatten(1).float()
        pred_flat = proposal_masks.flatten(1).float()
        intersection = gt_flat @ pred_flat.T
        union = gt_flat.sum(1)[:, None] + pred_flat.sum(1)[None] - intersection
        iou = intersection / union.clamp_min(1)
        valid = iou > 0.5
        # One extra valid match outweighs any possible gain in summed IoU.
        bonus = min(iou.shape) + 1
        quality = torch.where(valid, bonus + iou, 0)
        gt_indices, query_indices = linear_sum_assignment(-quality.cpu().numpy())
        accepted = valid[gt_indices, query_indices].cpu().numpy()
        gt_indices, query_indices = gt_indices[accepted], query_indices[accepted]
        if len(query_indices) == 0:
            return panoptic_map, []

        selected = proposals[torch.as_tensor(query_indices, device=proposals.device)]
        assigned_gt = gt_masks[torch.as_tensor(gt_indices, device=proposals.device)]
        while len(gt_indices):
            best_logits, winners = selected.max(dim=0)
            panoptic_map.zero_()
            panoptic_map[best_logits > 0] = winners[best_logits > 0] + 1
            final_ious = []
            for index, target in enumerate(assigned_gt):
                visible = (panoptic_map == index + 1) & (gt_map != 0)
                intersection = (visible & target).sum()
                union = (visible | target).sum()
                final_ious.append(intersection.float() / union.clamp_min(1))
            final_ious = torch.stack(final_ious)
            if (final_ious > 0.5).all():
                break
            # Remove the worst failed mask first: repainting can rescue others.
            keep = torch.arange(len(gt_indices), device=proposals.device) != final_ious.argmin()
            selected, assigned_gt = selected[keep], assigned_gt[keep]
            gt_indices = gt_indices[keep.cpu().numpy()]
        else:
            panoptic_map.zero_()
        segments = [
            {**gt_info[gt_index], "id": index + 1}
            for index, gt_index in enumerate(gt_indices)
        ]
        return panoptic_map, segments

    def _accumulate_image(self, gt_map, gt_info, pred_map, pred_info, stats=None):
        stats = self._stats if stats is None else stats
        gt_segments = {segment["id"]: segment for segment in gt_info}
        pred_segments = {segment["id"]: segment for segment in pred_info}
        gt_area = torch.bincount(gt_map.flatten())
        pred_area = torch.bincount(pred_map.flatten())
        pair_base = max(int(pred_map.max().item()) + 1, 1)
        intersections = torch.bincount((gt_map * pair_base + pred_map).flatten())

        matched_gt = set()
        matched_pred = set()
        for pair_id in torch.nonzero(intersections, as_tuple=False).flatten().tolist():
            gt_id, pred_id = divmod(pair_id, pair_base)
            if gt_id == 0 or pred_id == 0:
                continue
            gt_segment = gt_segments.get(gt_id)
            pred_segment = pred_segments.get(pred_id)
            if gt_segment is None or pred_segment is None:
                continue
            category_id = gt_segment["category_id"]
            if category_id != pred_segment["category_id"]:
                continue
            intersection = float(intersections[pair_id])
            void_overlap = float(intersections[pred_id]) if pred_id < intersections.numel() else 0.0
            union = float(gt_area[gt_id] + pred_area[pred_id]) - intersection - void_overlap
            iou = intersection / union if union > 0 else 0.0
            if iou > 0.5:
                stats[category_id, 0] += iou
                stats[category_id, 1] += 1
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)

        for gt_id, segment in gt_segments.items():
            if gt_id not in matched_gt:
                stats[segment["category_id"], 3] += 1
        for pred_id, segment in pred_segments.items():
            if pred_id in matched_pred:
                continue
            void_overlap = float(intersections[pred_id]) if pred_id < intersections.numel() else 0.0
            if pred_id < pred_area.numel() and void_overlap / max(float(pred_area[pred_id]), 1.0) > 0.5:
                continue
            stats[segment["category_id"], 2] += 1

    def evaluate(self):
        comm.synchronize()
        gathered = comm.gather(torch.stack((self._stats, self._oracle_stats)))
        if not comm.is_main_process():
            return None
        stats = torch.stack(gathered).sum(dim=0)
        results = OrderedDict({
            "panoptic_seg": self._summarize(stats[0]),
            "panoptic_seg_oracle": self._summarize(stats[1]),
        })
        rows = []
        for name, values in results.items():
            mode = "Oracle" if name.endswith("_oracle") else "Standard"
            for group, suffix in (("All", ""), ("Things", "_th"), ("Stuff", "_st")):
                rows.append([
                    mode, group,
                    *(f"{values[key + suffix]:.2f}" for key in ("PQ", "SQ", "RQ")),
                    *(values[key + suffix] for key in ("TP", "FP")),
                ])
        logging.getLogger(__name__).info(
            "Panoptic evaluation (PQ/SQ/RQ in percent; TP/FP are segment counts):\n%s",
            tabulate(
                rows,
                headers=("Evaluation", "Segments", "PQ", "SQ", "RQ", "TP", "FP"),
                tablefmt="github",
                disable_numparse=True,
            ),
        )
        return results

    def _summarize(self, stats):
        def metrics(category_ids):
            selected = stats[category_ids]
            iou, tp, fp, fn = selected.unbind(dim=1)
            denominator = tp + 0.5 * fp + 0.5 * fn
            valid = denominator > 0
            if not valid.any():
                return 0.0, 0.0, 0.0
            pq = (iou[valid] / denominator[valid]).mean()
            sq = torch.where(tp[valid] > 0, iou[valid] / tp[valid], 0).mean()
            rq = (tp[valid] / denominator[valid]).mean()
            return *(100.0 * value.item() for value in (pq, sq, rq)),

        pq, sq, rq = metrics(list(range(self.num_classes)))
        pq_th, sq_th, rq_th = metrics(list(range(1, self.num_classes)))
        pq_st, sq_st, rq_st = metrics([0])
        return {
            "PQ": pq, "SQ": sq, "RQ": rq,
            "PQ_th": pq_th, "SQ_th": sq_th, "RQ_th": rq_th,
            "PQ_st": pq_st, "SQ_st": sq_st, "RQ_st": rq_st,
            # Raw segment counts summed over images and distributed workers.
            "TP": int(stats[:, 1].sum().item()),
            "FP": int(stats[:, 2].sum().item()),
            "TP_th": int(stats[1:, 1].sum().item()),
            "FP_th": int(stats[1:, 2].sum().item()),
            "TP_st": int(stats[0, 1].item()),
            "FP_st": int(stats[0, 2].item()),
        }
