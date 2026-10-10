"""Compare the COCO adapter against official PQ, including void and crowd."""
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import numpy as np
from PIL import Image
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOPanopticEvaluator

from mask2former.evaluation.coco_panoptic_evaluation import OptimizedCOCOPanopticEvaluator


class COCOAdapterTest(unittest.TestCase):
    def test_matches_official_on_cpu_and_cuda(self):
        import multiprocessing
        from types import SimpleNamespace
        import panopticapi.evaluation as official_api
        original = official_api.multiprocessing
        official_api.multiprocessing = SimpleNamespace(cpu_count=lambda: 2, Pool=multiprocessing.Pool)
        self.addCleanup(setattr, official_api, "multiprocessing", original)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "coco_adapter_test_" + uuid4().hex
            self.addCleanup(MetadataCatalog.remove, name)
            categories = [{"id": 10, "name": "thing", "isthing": 1},
                          {"id": 30, "name": "stuff", "isthing": 0}]
            gt = np.array([[0, 1, 1, 2, 2], [0, 1, 1, 2, 2],
                           [3, 3, 4, 4, 4], [3, 3, 4, 4, 4]], dtype=np.int32)
            info = [{"id": 1, "category_id": 10, "iscrowd": 0, "area": 4},
                    {"id": 2, "category_id": 30, "iscrowd": 0, "area": 4},
                    {"id": 3, "category_id": 10, "iscrowd": 1, "area": 4},
                    {"id": 4, "category_id": 30, "iscrowd": 0, "area": 6}]
            rgb = np.stack((gt % 256, gt // 256 % 256, gt // 65536), axis=-1).astype(np.uint8)
            annotations = []
            for image_id in range(3):
                filename = f"{image_id}.png"
                Image.fromarray(rgb).save(root / filename)
                annotations.append({"image_id": image_id, "file_name": filename, "segments_info": info})
            annotation_file = root / "annotations.json"
            annotation_file.write_text(json.dumps({"categories": categories, "annotations": annotations}))
            MetadataCatalog.get(name).set(
                panoptic_root=str(root), panoptic_json=str(annotation_file),
                thing_dataset_id_to_contiguous_id={10: 0},
                stuff_dataset_id_to_contiguous_id={30: 1}, label_divisor=1000,
            )
            inputs, outputs = [], []
            for image_id in range(3):
                prediction = gt.copy()
                if image_id == 1:
                    prediction[prediction == 4] = 2  # merge two stuff segments
                if image_id == 2:
                    prediction[prediction == 1] = 0  # a false negative
                    prediction[prediction == 0] = 5  # a false positive over void + valid GT
                segments = []
                for identifier in np.unique(prediction):
                    if identifier == 0:
                        continue
                    thing = identifier in (1, 3, 5)
                    segments.append({"id": int(identifier), "category_id": 0 if thing else 1,
                                     "isthing": thing})
                inputs.append({"image_id": image_id, "file_name": f"{image_id}.jpg"})
                outputs.append({"panoptic_seg": (torch.from_numpy(prediction), segments)})
            official = COCOPanopticEvaluator(name)
            official.reset()
            official.process(inputs, outputs)
            expected = official.evaluate()["panoptic_seg"]
            for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
                with self.subTest(device=device):
                    evaluator = OptimizedCOCOPanopticEvaluator(name, device=device)
                    for _ in range(2):
                        evaluator.reset()
                        evaluator.process(inputs[:1], outputs[:1])
                        evaluator.process(inputs[1:], outputs[1:])
                        actual = evaluator.evaluate()["panoptic_seg"]
                        for key, value in expected.items():
                            self.assertAlmostEqual(actual[key], value, places=5)


if __name__ == "__main__":
    unittest.main()
