"""In-memory panoptic evaluation for the online synthetic-scene dataset."""

from collections import OrderedDict

import torch

from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm


class SyntheticScenePanopticEvaluator(DatasetEvaluator):
    """Compute standard PQ/SQ/RQ without serializing predictions or ground truth."""

    def __init__(self, num_classes=6):
        self.num_classes = num_classes

    def reset(self):
        # Columns are IoU sum, true positives, false positives, false negatives.
        self._stats = torch.zeros((self.num_classes, 4), dtype=torch.float64)

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

    def _accumulate_image(self, gt_map, gt_info, pred_map, pred_info):
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
                self._stats[category_id, 0] += iou
                self._stats[category_id, 1] += 1
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)

        for gt_id, segment in gt_segments.items():
            if gt_id not in matched_gt:
                self._stats[segment["category_id"], 3] += 1
        for pred_id, segment in pred_segments.items():
            if pred_id in matched_pred:
                continue
            void_overlap = float(intersections[pred_id]) if pred_id < intersections.numel() else 0.0
            if pred_id < pred_area.numel() and void_overlap / max(float(pred_area[pred_id]), 1.0) > 0.5:
                continue
            self._stats[segment["category_id"], 2] += 1

    def evaluate(self):
        comm.synchronize()
        gathered = comm.gather(self._stats)
        if not comm.is_main_process():
            return None
        stats = torch.stack(gathered).sum(dim=0)
        return OrderedDict({"panoptic_seg": self._summarize(stats)})

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
        }
