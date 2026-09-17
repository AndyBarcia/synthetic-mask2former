"""Batched, on-GPU input pipeline for the synthetic scene renderer."""

import importlib
import math
import sys
from pathlib import Path

import torch

import detectron2.utils.comm as comm
from detectron2.structures import BitMasks, Instances


_CLASS_ID_TO_TRAIN_ID = {2: 0, 10: 1, 11: 2, 12: 3, 13: 4, 14: 5}


def _synthetic_scene_module():
    """Import an installed renderer or the repository's in-place build."""
    try:
        return importlib.import_module("synthetic_scene")
    except ImportError:
        source_root = Path(__file__).resolve().parent / "datasets" / "synthetic-scene"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        try:
            return importlib.import_module("synthetic_scene")
        except ImportError as error:
            raise ImportError(
                "Build or install mask2former/data/datasets/synthetic-scene "
                "before using the synthetic dataset"
            ) from error


class SyntheticSceneBatchLoader:
    """Infinite iterable yielding already-batched Detectron2 model inputs."""

    def __init__(self, cfg, is_train=True):
        if not torch.cuda.is_available():
            raise RuntimeError("SyntheticSceneBatchLoader requires a CUDA device")

        world_size = comm.get_world_size()
        if cfg.SOLVER.IMS_PER_BATCH % world_size:
            raise ValueError("SOLVER.IMS_PER_BATCH must be divisible by the world size")
        self.is_train = is_train
        self.batch_size = (
            cfg.SOLVER.IMS_PER_BATCH // world_size
            if is_train
            else cfg.INPUT.SYNTHETIC_SCENE.TEST_BATCH_SIZE
        )
        self.rank = comm.get_rank()
        self.world_size = world_size

        options = cfg.INPUT.SYNTHETIC_SCENE
        self.width = options.WIDTH
        self.height = options.HEIGHT
        self.base_seed = options.SEED if is_train else options.TEST_SEED
        self.test_samples = options.TEST_SAMPLES
        self.panoptic = cfg.INPUT.DATASET_MAPPER_NAME == "synthetic_scene_panoptic_batch"
        renderer = _synthetic_scene_module()
        self.scene_options = renderer.RandomSceneOptions(
            house_count=options.HOUSE_COUNT,
            tree_count=options.TREE_COUNT,
            cloud_count=options.CLOUD_COUNT,
            car_count=options.CAR_COUNT,
            person_count=options.PERSON_COUNT,
            aspect_ratio=self.width / self.height,
        )

    def __iter__(self):
        render_random_scene = _synthetic_scene_module().render_random_scene

        iteration = 0
        while self.is_train or iteration < len(self):
            # Each distributed rank and iteration receives a disjoint renderer
            # seed while retaining one batched generator/render launch.
            batch_index = iteration * self.world_size + self.rank
            if self.is_train:
                current_batch_size = self.batch_size
                seed = self.base_seed + batch_index
            else:
                global_start = iteration * self.world_size * self.batch_size + self.rank * self.batch_size
                current_batch_size = min(self.batch_size, self.test_samples - global_start)
                if current_batch_size <= 0:
                    break
                seed = self.base_seed + global_start
            result = render_random_scene(
                seed,
                width=self.width,
                height=self.height,
                batch_size=current_batch_size,
                scene_options=self.scene_options,
                return_maps=True,
            )
            images = result.image.mul(255.0)
            raw_labels = result.semantic_map
            instance_maps = result.instance_map
            sem_seg = torch.full_like(raw_labels, 255, dtype=torch.long)
            for renderer_id, train_id in _CLASS_ID_TO_TRAIN_ID.items():
                sem_seg[raw_labels == renderer_id] = train_id

            yield [
                self._model_input(
                    images[index],
                    sem_seg[index],
                    instance_maps[index],
                    result.visible_classes[index],
                    result.visible_instance_ids[index],
                    seed,
                    index,
                )
                for index in range(current_batch_size)
            ]
            iteration += 1

    def __len__(self):
        if self.is_train:
            raise TypeError("the training loader is infinite")
        samples_for_rank = max(0, self.test_samples - self.rank * self.batch_size)
        return math.ceil(samples_for_rank / (self.world_size * self.batch_size))

    def _model_input(
        self,
        image,
        sem_seg,
        instance_map,
        visible_classes,
        visible_instance_ids,
        batch_seed,
        index,
    ):
        instances = Instances((self.height, self.width))
        if self.panoptic:
            valid = visible_instance_ids > 0
            thing_ids = visible_instance_ids[valid]
            renderer_classes = visible_classes[valid]
            thing_classes = torch.full_like(renderer_classes, -1, dtype=torch.long)
            for renderer_id, train_id in _CLASS_ID_TO_TRAIN_ID.items():
                thing_classes[renderer_classes == renderer_id] = train_id
            # Class 0 is terrain. The renderer may report its positive custom
            # instance ID in visible metadata, but panoptic targets keep it only
            # as the single stuff mask constructed below.
            valid_things = thing_classes > 0
            thing_ids = thing_ids[valid_things]
            thing_classes = thing_classes[valid_things]

            # Terrain is the sole stuff segment and uses class 0. Background is
            # excluded even though both it and terrain have no custom instance ID.
            terrain_mask = sem_seg == 0
            has_terrain = terrain_mask.any().reshape(1)
            instances.gt_classes = torch.cat(
                (
                    torch.zeros((1,), dtype=torch.long, device=image.device)[has_terrain],
                    thing_classes,
                )
            )
            thing_masks = (instance_map.unsqueeze(0) == thing_ids[:, None, None]) & (
                sem_seg.unsqueeze(0) == thing_classes[:, None, None]
            )
            instances.gt_masks = torch.cat(
                (terrain_mask.unsqueeze(0)[has_terrain], thing_masks), dim=0
            )
        else:
            classes = torch.unique(sem_seg)
            classes = classes[classes != 255]
            instances.gt_classes = classes
            if classes.numel():
                instances.gt_masks = BitMasks(
                    sem_seg.unsqueeze(0) == classes[:, None, None]
                ).tensor
            else:
                instances.gt_masks = torch.zeros(
                    (0, self.height, self.width), dtype=torch.bool, device=image.device
                )
        record = {
            "image": image.contiguous(),
            "sem_seg": sem_seg.contiguous(),
            "instances": instances,
            "height": self.height,
            "width": self.width,
            "image_id": f"{batch_seed}:{index}",
        }
        if self.panoptic and not self.is_train:
            panoptic_seg = torch.zeros_like(sem_seg, dtype=torch.int32)
            segments_info = []
            for segment_index, (category_id, mask) in enumerate(
                zip(instances.gt_classes.tolist(), instances.gt_masks), start=1
            ):
                panoptic_seg[mask] = segment_index
                segments_info.append(
                    {
                        "id": segment_index,
                        "category_id": category_id,
                        "isthing": category_id > 0,
                    }
                )
            record["panoptic_ground_truth"] = (panoptic_seg, segments_info)
        return record
