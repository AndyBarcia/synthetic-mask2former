"""Deterministic square validation matching the dense-mask training geometry."""
import copy
import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


class YOLOXPanopticTestMapper:
    """Resize the long edge to PAD_SIZE, pad with raw value 128, track valid size.

    Annotations are not read. The model crops padding before returning original
    resolution predictions. This reproduces the proposal experiments' validation
    geometry without using a training mapper or exposing GT to inference.
    """

    def __init__(self, cfg):
        self.size = cfg.MODEL.YOLOX.PAD_SIZE
        self.format = cfg.INPUT.FORMAT

    def __call__(self, record):
        record = copy.deepcopy(record)
        image = utils.read_image(record['file_name'], format=self.format)
        utils.check_image_size(record, image)
        resize = T.ResizeScale(1., 1., self.size, self.size).get_transform(image)
        image = resize.apply_image(image)
        h, w = image.shape[:2]
        record['valid_resized_size'] = (h, w)
        image = np.pad(image, ((0, self.size - h), (0, self.size - w), (0, 0)), constant_values=128)
        record['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        record.pop('annotations', None)
        return record
