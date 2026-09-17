"""In-memory semantic evaluation for the online synthetic-scene dataset."""

from collections import OrderedDict

import torch

from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

from mask2former.data.datasets.register_synthetic_scene import SYNTHETIC_SCENE_CLASSES


class SyntheticSceneSemSegEvaluator(DatasetEvaluator):
    """Compute semantic metrics directly from generated targets and logits."""

    def __init__(self, class_names=None, ignore_label=255):
        self.class_names = class_names or SYNTHETIC_SCENE_CLASSES
        self.num_classes = len(self.class_names)
        self.ignore_label = ignore_label

    def reset(self):
        size = self.num_classes + 1
        self._confusion = torch.zeros((size, size), dtype=torch.int64)

    def process(self, inputs, outputs):
        size = self.num_classes + 1
        for input_record, output_record in zip(inputs, outputs):
            prediction = output_record["sem_seg"].argmax(dim=0).to(
                device="cpu", dtype=torch.int64
            )
            target = input_record["sem_seg"].detach().to(device="cpu", dtype=torch.int64)
            target[target == self.ignore_label] = self.num_classes
            indices = size * prediction.flatten() + target.flatten()
            self._confusion += torch.bincount(
                indices, minlength=size * size
            ).reshape(size, size)

    def evaluate(self):
        comm.synchronize()
        gathered = comm.gather(self._confusion)
        if not comm.is_main_process():
            return None
        confusion = torch.stack(gathered).sum(dim=0).to(torch.float64)
        return OrderedDict({"sem_seg": self._summarize(confusion)})

    def _summarize(self, confusion):
        true_positive = confusion.diagonal()[:-1]
        ground_truth = confusion[:-1, :-1].sum(dim=0)
        predicted = confusion[:-1, :-1].sum(dim=1)
        union = ground_truth + predicted - true_positive

        accuracy = torch.full((self.num_classes,), torch.nan, dtype=torch.float64)
        iou = torch.full_like(accuracy, torch.nan)
        accuracy_valid = ground_truth > 0
        iou_valid = accuracy_valid & (union > 0)
        accuracy[accuracy_valid] = true_positive[accuracy_valid] / ground_truth[accuracy_valid]
        iou[iou_valid] = true_positive[iou_valid] / union[iou_valid]

        total_ground_truth = ground_truth.sum()
        weights = ground_truth / total_ground_truth if total_ground_truth > 0 else ground_truth
        result = {
            "mIoU": 100.0 * iou[iou_valid].mean().item() if iou_valid.any() else 0.0,
            "fwIoU": 100.0 * (iou[iou_valid] * weights[iou_valid]).sum().item(),
            "mACC": 100.0 * accuracy[accuracy_valid].mean().item()
            if accuracy_valid.any()
            else 0.0,
            "pACC": 100.0 * (true_positive.sum() / total_ground_truth).item()
            if total_ground_truth > 0
            else 0.0,
        }
        for index, name in enumerate(self.class_names):
            result[f"IoU-{name}"] = 100.0 * iou[index].item()
            result[f"ACC-{name}"] = 100.0 * accuracy[index].item()
        return result
