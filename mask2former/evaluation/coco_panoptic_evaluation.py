"""COCO PQ evaluation through the in-memory panoptic-evaluator backend."""
from collections import OrderedDict
import json
import logging
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from .combined_evaluation import instance_batch, instance_state, summarize_instances
from .synthetic_scene_panoptic_evaluation import SyntheticScenePanopticEvaluator

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager


class OptimizedCOCOPanopticEvaluator(DatasetEvaluator):
    """Read original COCO targets and accumulate PQ without prediction PNGs."""

    def __init__(self, dataset_name, output_dir=None, device=None, evaluate_instance=False, evaluate_semantic=False):
        self.metadata = MetadataCatalog.get(dataset_name)
        self.output_dir = output_dir
        self.evaluate_instance = evaluate_instance
        self.evaluate_semantic = evaluate_semantic
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with PathManager.open(self.metadata.panoptic_json) as handle:
            annotations = json.load(handle)
        self.targets = {ann["image_id"]: ann for ann in annotations["annotations"]}
        self.class_names = [category.get("name", str(category["id"])) for category in annotations["categories"]]
        self.category_ids = [category["id"] for category in annotations["categories"]]
        self.num_classes = len(self.category_ids)
        self.isthing = [bool(category["isthing"]) for category in annotations["categories"]]
        self.thing_ids = {v: k for k, v in self.metadata.thing_dataset_id_to_contiguous_id.items()}
        self.stuff_ids = {v: k for k, v in self.metadata.stuff_dataset_id_to_contiguous_id.items()}
        submodule = Path(__file__).resolve().parents[2] / "panoptic-evaluator"
        if str(submodule) not in sys.path:
            sys.path.insert(0, str(submodule))
        from panoptic_evaluator import PanopticBatch, PanopticEvaluator, SegmentationEvaluator, InstanceBatch
        self.batch_type = PanopticBatch
        self.evaluator_type = SegmentationEvaluator if evaluate_instance else PanopticEvaluator
        self.instance_batch_type = InstanceBatch
        self.instance_classes = {label: self.category_ids.index(category)
                                 for label, category in self.thing_ids.items()}

    def reset(self):
        self.metrics = self.evaluator_type(
            len(self.category_ids), isthing=self.isthing, device=self.device,
        )
        self.image_count = 0

    def process(self, inputs, outputs):
        groups = {}
        for record, output in zip(inputs, outputs):
            target = self.targets[record["image_id"]]
            filename = str(Path(self.metadata.panoptic_root) / target["file_name"])
            with PathManager.open(filename, "rb") as handle:
                rgb = np.asarray(Image.open(handle).convert("RGB"), dtype=np.int32)
            gt_map = torch.from_numpy(rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2])
            pred_map, segments = output["panoptic_seg"]
            pred_map = pred_map.detach().to(self.device, dtype=torch.int32)
            if segments is None:
                # Detectron2 also accepts category * label_divisor + instance maps,
                # with -1 as void; shift them to reserve segment ID zero.
                segments = []
                for identifier in torch.unique(pred_map).cpu().tolist():
                    if identifier == -1:
                        continue
                    category = identifier // self.metadata.label_divisor
                    segments.append({"id": identifier + 1, "category_id": category,
                                     "isthing": category in self.thing_ids})
                pred_map = pred_map + 1
            infos = []
            for segment in segments:
                info = dict(segment)
                isthing = info.pop("isthing", None)
                if isthing is not None:
                    mapping = self.thing_ids if isthing else self.stuff_ids
                    info["category_id"] = mapping[info["category_id"]]
                infos.append(info)
            if tuple(pred_map.shape) != tuple(gt_map.shape):
                raise ValueError("COCO panoptic predictions must match original image dimensions")
            groups.setdefault(tuple(gt_map.shape), []).append(
                (gt_map, target["segments_info"], pred_map, infos, output))
        for entries in groups.values():
            gt_maps, gt_infos, pred_maps, pred_infos, predictions = zip(*entries)
            target = self.batch_type.from_coco(
                torch.stack(gt_maps).to(self.device), gt_infos, self.category_ids, compact=False,
            )
            prediction = self.batch_type.from_coco(
                torch.stack(pred_maps), pred_infos, self.category_ids, compact=False,
            )
            if self.evaluate_instance:
                areas = torch.zeros_like(target.classes, dtype=torch.float64)
                for row, info in enumerate(gt_infos):
                    areas[row, 1:len(info) + 1] = torch.tensor(
                        [segment["area"] for segment in info], device=self.device)
                self.metrics.update(prediction, instance_batch(
                    predictions, self.instance_batch_type, self.device, self.instance_classes),
                    target, instance_areas=areas)
            else:
                self.metrics.update(prediction, target)
            self.image_count += len(entries)

    def evaluate(self):
        comm.synchronize()
        panoptic = self.metrics.panoptic if self.evaluate_instance else self.metrics
        stats = torch.stack((panoptic.iou_sum, panoptic.tp,
                             panoptic.fp, panoptic.fn), dim=1).double().cpu()
        gathered = comm.gather((stats, self.image_count, panoptic.confusion.cpu(),
                                instance_state(self.metrics.instance) if self.evaluate_instance else None))
        if not comm.is_main_process():
            return None
        if not sum(count for _, count, _, _ in gathered):
            return {}
        stats = torch.stack([values for values, _, _, _ in gathered]).sum(dim=0)
        result = {}
        for suffix, selected in (("", list(range(len(self.category_ids)))),
                                 ("_th", [i for i, thing in enumerate(self.isthing) if thing]),
                                 ("_st", [i for i, thing in enumerate(self.isthing) if not thing])):
            iou, tp, fp, fn = stats[selected].unbind(dim=1)
            denominator = tp + 0.5 * (fp + fn)
            valid = denominator > 0
            values = (iou / denominator.clamp_min(1),
                      iou / tp.clamp_min(1), tp / denominator.clamp_min(1))
            for name, value in zip(("PQ", "SQ", "RQ"), values):
                result[name + suffix] = 100 * value[valid].mean().item() if valid.any() else 0.0
        logging.getLogger(__name__).info("COCO panoptic evaluation (optimized API): %s", result)
        if self.output_dir:
            PathManager.mkdirs(self.output_dir)
            with PathManager.open(str(Path(self.output_dir) / "panoptic_metrics.json"), "w") as handle:
                json.dump(result, handle, indent=2)
        results = OrderedDict(panoptic_seg=result)
        if self.evaluate_semantic:
            confusion = torch.stack([item[2] for item in gathered]).sum(0)
            results["sem_seg"] = SyntheticScenePanopticEvaluator._summarize_semantic(self, confusion)
        if self.evaluate_instance:
            results["segm"] = summarize_instances([item[3] for item in gathered], len(self.category_ids), self.device)
        return results
