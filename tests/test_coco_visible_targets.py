"""Discard vanished annotations without rejecting training views."""
from unittest.mock import patch
import numpy as np
from mask2former.data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper


def check():
    from types import SimpleNamespace
    image=np.zeros((4,4,3),dtype=np.uint8)
    panoptic=np.zeros_like(image)
    panoptic[:,0,0]=1;panoptic[:,1,0]=3;panoptic[:,2:,0]=2
    record=dict(file_name='image',pan_seg_file_name='panoptic',height=4,width=4,image_id=17,
        segments_info=[dict(id=1,category_id=0,iscrowd=False),
                       dict(id=2,category_id=1,iscrowd=False),dict(id=3,category_id=2,iscrowd=True)])
    def run(mapper,crop):
        transforms=SimpleNamespace(apply_segmentation=lambda x:crop(x))
        with patch('detectron2.data.detection_utils.read_image',side_effect=lambda path,format: image if path=='image' else panoptic), \
             patch('detectron2.data.transforms.apply_transform_gens',return_value=(crop(image),transforms)):
            return mapper(record)
    mapper=COCOPanopticNewBaselineDatasetMapper(tfm_gens=[],image_format='RGB')
    cropped=run(mapper,lambda x:x[:,:2])
    assert cropped['instances'].gt_classes.tolist()==[0]
    assert cropped['instances'].gt_masks.shape==(1,4,2)
    assert cropped['instances'].gt_masks.sum().item()==4
    # Views are never resampled for target count; the criterion limits only
    # optional unions to the remaining slots.
    assert run(mapper,lambda x:x)['instances'].gt_classes.tolist()==[0,1]
    # Every target may vanish after augmentation; the empty image remains valid.
    empty=run(mapper,lambda x:np.zeros_like(x))
    assert len(empty['instances'])==0 and empty['instances'].gt_masks.shape==(0,4,4)
    print('cropped targets preserved without resampling; empty scenes PASS')


if __name__=='__main__':
    check()
