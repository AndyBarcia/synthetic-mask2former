"""In-memory panoptic evaluation for the online synthetic-scene dataset."""

from collections import OrderedDict
import logging
from pathlib import Path
import sys

import torch
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm


class SyntheticScenePanopticEvaluator(DatasetEvaluator):
    """Compute standard PQ/SQ/RQ without serializing predictions or ground truth."""

    def __init__(self, num_classes=6, device=None, class_names=None, evaluate_oracle=True):
        self.num_classes = num_classes
        self.evaluate_oracle = evaluate_oracle
        self.class_names = class_names or [str(index) for index in range(num_classes)]
        if len(self.class_names) != num_classes:
            raise ValueError("class_names must match num_classes")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Use the pinned submodule directly, including its in-place CUDA build.
        submodule = Path(__file__).resolve().parents[2] / "panoptic-evaluator"
        if not (submodule / "panoptic_evaluator").is_dir():
            raise ImportError("Initialize the evaluator with git submodule update --init --recursive")
        if str(submodule) not in sys.path:
            sys.path.insert(0, str(submodule))
        from panoptic_evaluator import PanopticBatch, PanopticEvaluator
        self._batch_type = PanopticBatch
        self._evaluator_type = PanopticEvaluator

    def reset(self):
        options = dict(isthing=[False] + [True] * (self.num_classes - 1), device=self.device)
        self._standard = self._evaluator_type(self.num_classes, **options)
        self._oracle = self._evaluator_type(self.num_classes, **options)

    @staticmethod
    def _raw_stats(evaluator):
        # Preserve the existing distributed reduction and metric schema.
        return torch.stack((evaluator.iou_sum, evaluator.tp,
                            evaluator.fp, evaluator.fn), dim=1).double().cpu()

    @property
    def _stats(self):
        return self._raw_stats(self._standard)

    @property
    def _oracle_stats(self):
        return self._raw_stats(self._oracle)

    def process(self, inputs, outputs):
        # Uniform canvases share one native evaluator update, not one per image.
        groups = {}
        for index, record in enumerate(inputs):
            groups.setdefault(tuple(record["panoptic_ground_truth"][0].shape), []).append(index)
        for indices in groups.values():
            records = [inputs[index] for index in indices]
            predictions = [outputs[index] for index in indices]
            target = self._pack_ground_truth(records)
            self._accumulate_batch(predictions, target, self._standard)
            if self.evaluate_oracle:
                oracle = [self._oracle_assignment(record["panoptic_ground_truth"][0],
                                                   record["panoptic_ground_truth"][1],
                                                   prediction["panoptic_proposals"])
                          for record, prediction in zip(records, predictions)]
                self._accumulate_batch([{"panoptic_seg": result} for result in oracle], target, self._oracle)

    def _pack_ground_truth(self, records):
        maps, infos = zip(*(record["panoptic_ground_truth"] for record in records))
        return self._batch_type.from_coco(
            torch.stack(maps).to(device=self.device, dtype=torch.int64),
            infos, list(range(self.num_classes)), compact=False)

    def _accumulate_batch(self, predictions, target, evaluator):
        maps, infos = zip(*(prediction["panoptic_seg"] for prediction in predictions))
        maps = torch.stack(maps).to(device=self.device, dtype=torch.int32)
        if all("panoptic_classes" in prediction for prediction in predictions):
            classes = torch.nn.utils.rnn.pad_sequence(
                [prediction["panoptic_classes"].to(self.device) for prediction in predictions],
                batch_first=True, padding_value=-1)
            prediction = self._batch_type(maps, classes)
        else:
            prediction = self._batch_type.from_coco(
                maps, infos, list(range(self.num_classes)), compact=False)
        evaluator.update(prediction, target)

    @staticmethod
    def _oracle_assignment(gt_map, gt_info, proposals):
        """Prioritize the number of one-to-one matches with binary IoU > 0.5.

        Classes and proposal selection are oracle decisions. Selected masks keep
        their predicted geometry; overlapping pixels go to the highest logit,
        and pixels outside every selected binary mask remain void. Prune masks
        that fail the same strict threshold after overlap resolution, repainting
        after each removal. This is a heuristic, not a global PQ optimization.
        """
        if hasattr(proposals, "materialize"):
            proposals = proposals.materialize()
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
            # Score all assigned masks in one reduction instead of launching
            # several kernels per GT segment during every pruning round.
            ids = torch.arange(1, len(gt_indices) + 1, device=proposals.device)
            visible = (panoptic_map[None] == ids[:, None, None]) & (gt_map != 0)[None]
            intersection = (visible & assigned_gt).flatten(1).sum(1)
            union = visible.flatten(1).sum(1) + assigned_gt.flatten(1).sum(1) - intersection
            final_ious = intersection.float() / union.clamp_min(1)
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
        evaluator = self._standard if stats is None else stats
        categories = list(range(self.num_classes))
        target = self._batch_type.from_coco(
            gt_map.detach().to(device=self.device, dtype=torch.int64)[None],
            [gt_info], categories, compact=False,
        )
        prediction = self._batch_type.from_coco(
            pred_map.detach().to(device=self.device, dtype=torch.int64)[None],
            [pred_info], categories, compact=False,
        )
        evaluator.update(prediction, target)

    def evaluate(self):
        comm.synchronize()
        gathered = comm.gather((
            torch.stack((self._stats, self._oracle_stats)),
            self._standard.confusion.cpu(),
        ))
        if not comm.is_main_process():
            return None
        stats = torch.stack([item[0] for item in gathered]).sum(dim=0)
        confusion = torch.stack([item[1] for item in gathered]).sum(dim=0)
        results = OrderedDict({"panoptic_seg": self._summarize(stats[0])})
        if self.evaluate_oracle:
            results["panoptic_seg_oracle"] = self._summarize(stats[1])
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
        results["sem_seg"] = self._summarize_semantic(confusion)
        logging.getLogger(__name__).info(
            "Semantic evaluation from panoptic predictions (percent): %s",
            {key: round(results["sem_seg"][key], 4)
             for key in ("mIoU", "fwIoU", "mACC", "pACC")},
        )
        return results

    def _summarize_semantic(self, confusion):
        # Backend rows are GT classes; columns are predicted classes plus void.
        # GT void/crowd pixels are excluded. Predicted void contributes to FN.
        confusion = confusion.double()
        tp = confusion[:, :self.num_classes].diagonal()
        ground_truth = confusion.sum(dim=1)
        predicted = confusion[:, :self.num_classes].sum(dim=0)
        union = ground_truth + predicted - tp
        valid_iou = union > 0
        valid_accuracy = ground_truth > 0
        iou = torch.where(valid_iou, tp / union.clamp_min(1), torch.nan)
        accuracy = torch.where(valid_accuracy, tp / ground_truth.clamp_min(1), torch.nan)
        total = ground_truth.sum()
        result = {
            "mIoU": 100.0 * iou[valid_iou].mean().item() if valid_iou.any() else 0.0,
            "fwIoU": 100.0 * (torch.nan_to_num(iou) * ground_truth).sum().item()
            / total.clamp_min(1).item(),
            "mACC": 100.0 * accuracy[valid_accuracy].mean().item()
            if valid_accuracy.any() else 0.0,
            "pACC": 100.0 * tp.sum().item() / total.clamp_min(1).item(),
        }
        for index, name in enumerate(self.class_names):
            result[f"IoU-{name}"] = 100.0 * iou[index].item()
            result[f"ACC-{name}"] = 100.0 * accuracy[index].item()
        return result

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
