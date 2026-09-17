"""Register deterministic seed streams for the online synthetic renderer."""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog


SYNTHETIC_SCENE_CLASSES = ["terrain", "house", "tree", "cloud", "car", "person"]
SYNTHETIC_SCENE_COLORS = [
    [92, 117, 76],
    [176, 112, 72],
    [41, 132, 54],
    [215, 225, 235],
    [196, 38, 32],
    [44, 86, 196],
]


def _seed_records(size, seed_offset):
    # Keep catalog entries tiny; the batch loader owns actual generation.
    return [
        {"image_id": index, "synthetic_scene_seed": seed_offset + index}
        for index in range(size)
    ]


def register_synthetic_scene(
    name,
    *,
    size,
    seed_offset=0,
):
    """Register an on-demand synthetic dataset backed only by integer seeds."""
    if size <= 0:
        raise ValueError("synthetic dataset size must be positive")
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(
            name,
            lambda size=size, seed_offset=seed_offset: _seed_records(size, seed_offset),
        )
    MetadataCatalog.get(name).set(
        stuff_classes=SYNTHETIC_SCENE_CLASSES,
        stuff_colors=SYNTHETIC_SCENE_COLORS,
        evaluator_type="synthetic_scene_sem_seg",
        ignore_label=255,
    )


def register_synthetic_scene_panoptic(name, *, size, seed_offset=0):
    """Register an online panoptic split with terrain as its only stuff class."""
    if size <= 0:
        raise ValueError("synthetic dataset size must be positive")
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(
            name,
            lambda size=size, seed_offset=seed_offset: _seed_records(size, seed_offset),
        )
    MetadataCatalog.get(name).set(
        thing_classes=SYNTHETIC_SCENE_CLASSES[1:],
        thing_colors=SYNTHETIC_SCENE_COLORS[1:],
        stuff_classes=[SYNTHETIC_SCENE_CLASSES[0]],
        stuff_colors=[SYNTHETIC_SCENE_COLORS[0]],
        thing_dataset_id_to_contiguous_id={
            renderer_id: train_id
            for renderer_id, train_id in zip(range(10, 15), range(1, 6))
        },
        stuff_dataset_id_to_contiguous_id={2: 0},
        evaluator_type="synthetic_scene_panoptic_seg",
        ignore_label=255,
    )


def register_all_synthetic_scene():
    train_size = int(os.getenv("SYNTHETIC_SCENE_TRAIN_SIZE", "10000"))
    val_size = int(os.getenv("SYNTHETIC_SCENE_VAL_SIZE", "1000"))
    base_seed = int(os.getenv("SYNTHETIC_SCENE_SEED", "1234"))
    register_synthetic_scene(
        "synthetic_scene_sem_seg_train", size=train_size, seed_offset=base_seed
    )
    register_synthetic_scene_panoptic(
        "synthetic_scene_panoptic_train", size=train_size, seed_offset=base_seed
    )
    register_synthetic_scene_panoptic(
        "synthetic_scene_panoptic_val",
        size=val_size,
        seed_offset=base_seed + train_size,
    )
    register_synthetic_scene(
        "synthetic_scene_sem_seg_val",
        size=val_size,
        seed_offset=base_seed + train_size,
    )


register_all_synthetic_scene()
