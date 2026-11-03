"""MaskFormer-style meta-architecture with dense YOLOX mask proposals."""
import torch
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.modeling.postprocessing import sem_seg_postprocess

from .maskformer_model import MaskFormer
from .modeling.yolox.criterion import YOLOXCriterion
from .modeling.yolox.matcher import YOLOXMatcher
from .modeling.yolox.object_decoder import ProposalObjectDecoder
from .modeling.panoptic_paint import paint_panoptic_masks, score_panoptic_masks


@META_ARCH_REGISTRY.register()
class YOLOXObjectDecoder(MaskFormer):
    """Reuse MaskFormer's preprocessing, targets, loss weighting, and trainer contract."""

    @configurable
    def __init__(self, *, backbone, sem_seg_head, criterion, metadata, pixel_mean, pixel_std,
                 num_queries, max_steps, rendering, freeze_backbone=False, freeze_pixel_decoder=False,
                 freeze_memory=False, freeze_proposals=False, semantic_on=False, instance_on=False,
                 size_divisibility=32, test_topk_per_image=100):
        super().__init__(
            backbone=backbone, sem_seg_head=sem_seg_head, criterion=criterion, num_queries=num_queries,
            object_mask_threshold=0., panoptic_paint_order='forward', overlap_threshold=0., metadata=metadata,
            size_divisibility=size_divisibility, sem_seg_postprocess_before_inference=True,
            pixel_mean=pixel_mean, pixel_std=pixel_std, semantic_on=semantic_on, panoptic_on=True,
            instance_on=instance_on, test_topk_per_image=test_topk_per_image,
            object_rl_only=criterion.object_rl_only)
        self.max_steps = max_steps
        self.rendering = rendering
        self.freeze_backbone = freeze_backbone or self.object_rl_only
        self.freeze_pixel_decoder = freeze_pixel_decoder or self.object_rl_only
        self.freeze_memory = freeze_memory or self.object_rl_only
        self.freeze_proposals = freeze_proposals or self.object_rl_only
        for module, frozen in self._freeze_modules():
            module.requires_grad_(not frozen)
        self.sem_seg_head.token_adapter.requires_grad_(True)
        self.train(self.training)

    @classmethod
    def from_config(cls, cfg):
        y, m = cfg.MODEL.YOLOX, cfg.MODEL.MASK_FORMER
        if y.TOPK <= 0 or y.MAX_STEPS <= 0 or y.WIDTH <= 0 or y.WIDTH % 8:
            raise ValueError('YOLOX budgets must be positive and WIDTH divisible by eight')
        if not 0 <= y.MATCH_IOU_THRESHOLD <= 1 or y.RENDERING not in ('forward', 'pixel-score'):
            raise ValueError('Invalid object matching IoU or renderer')
        if y.RL_SAMPLES < 2:
            raise ValueError('RLOO requires at least two samples')
        if y.OBJECT_DEC_SUPERVISION not in ("hungarian", "dynamic_k_grouped"):
            raise ValueError("Invalid YOLOX object decoder supervision")
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        things = metadata.thing_dataset_id_to_contiguous_id.values()
        matcher = YOLOXMatcher(sem_seg_head.num_classes, things, box_free=y.BOX_FREE,
                               soft_labels=y.SOFT_LABELS, object_iou_threshold=y.MATCH_IOU_THRESHOLD,
                               cost_class=m.CLASS_WEIGHT, cost_mask=m.MASK_WEIGHT,
                               cost_dice=m.DICE_WEIGHT, num_points=m.TRAIN_NUM_POINTS,
                               ensure_gt_coverage=y.OBJECT_DEC_SUPERVISION == "dynamic_k_grouped")
        weight_dict = {
            'loss_objectness': y.OBJECTNESS_WEIGHT, 'loss_ce': m.CLASS_WEIGHT,
            'loss_mask': m.MASK_WEIGHT, 'loss_dice': m.DICE_WEIGHT,
            'loss_stuff_presence': y.STUFF_PRESENCE_WEIGHT,
            'loss_stuff_mask': m.MASK_WEIGHT, 'loss_stuff_dice': m.DICE_WEIGHT,
            'loss_object_decoder': y.SEQUENCE_LOSS_WEIGHT,
        }
        if not y.BOX_FREE:
            weight_dict['loss_box'] = y.BOX_WEIGHT
        if m.OBJECT_RL_WEIGHT > 0:
            weight_dict['loss_object_rl'] = m.OBJECT_RL_WEIGHT
        criterion = YOLOXCriterion(
            sem_seg_head.num_classes, matcher, weight_dict,
            ProposalObjectDecoder(cfg.MODEL.SEM_SEG_HEAD.MASK_DIM, m.HIDDEN_DIM, y.TOPK,
                                  m.OBJECT_DEC_LAYERS, m.NHEADS, m.DIM_FEEDFORWARD,
                                  mask_iou_threshold=1., prefix_tree_branches=m.OBJECT_DEC_PREFIX_TREE_BRANCHES,
                                  position_encoding=m.OBJECT_DEC_POSITION_ENCODING, rope=m.OBJECT_DEC_ROPE,
                                  eof_threshold=m.OBJECT_DEC_EOF_THRESHOLD, collapse_masks=False),
            things, full_mask_loss=y.FULL_MASK_LOSS, object_rl_weight=m.OBJECT_RL_WEIGHT,
            object_dec_supervision=y.OBJECT_DEC_SUPERVISION,
            object_rl_only=m.OBJECT_RL_ONLY, object_rl_num_samples=y.RL_SAMPLES,
            object_rl_max_steps=y.MAX_STEPS, rendering=y.RENDERING)
        return dict(
            backbone=backbone, sem_seg_head=sem_seg_head, criterion=criterion, metadata=metadata,
            pixel_mean=cfg.MODEL.PIXEL_MEAN, pixel_std=cfg.MODEL.PIXEL_STD,
            num_queries=y.TOPK, max_steps=y.MAX_STEPS, rendering=y.RENDERING,
            freeze_backbone=y.FREEZE_BACKBONE, freeze_pixel_decoder=y.FREEZE_PIXEL_DECODER,
            freeze_memory=y.FREEZE_MEMORY, freeze_proposals=y.FREEZE_PROPOSALS,
            semantic_on=m.TEST.SEMANTIC_ON, instance_on=m.TEST.INSTANCE_ON,
            size_divisibility=m.SIZE_DIVISIBILITY, test_topk_per_image=cfg.TEST.DETECTIONS_PER_IMAGE)

    def _freeze_modules(self):
        return [(self.backbone, self.freeze_backbone),
                (self.sem_seg_head.pixel_decoder, self.freeze_pixel_decoder),
                (self.sem_seg_head.memory, self.freeze_memory),
                (self.sem_seg_head.predictor, self.freeze_proposals)]

    def train(self, mode=True):
        super().train(mode)
        for module, frozen in self._freeze_modules():
            if frozen:
                module.eval()
        return self

    def forward(self, batched_inputs):
        images = [(x['image'].to(self.device) - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, padded_size=tuple(images.tensor.shape[-2:]))
        if self.training:
            instances = [x['instances'].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(instances, images)
            losses = self.criterion(outputs, targets)
            for key in list(losses):
                if key in self.criterion.weight_dict:
                    losses[key] *= self.criterion.weight_dict[key]
                else:
                    losses.pop(key)
            return losses
        orders, _ = self.criterion.object_decoder.rollout(
            outputs['mask_embeddings'], outputs['object_decoder_image_features'], outputs['pred_masks'],
            outputs['object_decoder_image_sizes'], sample=False, max_steps=self.max_steps)
        return [self._inference(p, order, item, resized, outputs['padded_size'])
                for p, order, item, resized in zip(outputs['proposals'], orders, batched_inputs, images.image_sizes)]

    def _inference(self, proposal, order, item, resized, padded_size):
        ids = torch.as_tensor(order, device=self.device, dtype=torch.long)
        masks = proposal['masks'][ids]
        height, width = item.get('height', resized[0]), item.get('width', resized[1])
        if len(ids):
            masks = F.interpolate(masks[None].float(), padded_size, mode='bilinear', align_corners=False)[0]
            masks = sem_seg_postprocess(masks, resized, height, width)
        else:
            masks = masks.new_empty((0, height, width))
        labels, scores = proposal['labels'][ids], proposal['scores'][ids]
        valid = torch.ones((1, len(ids)), device=self.device, dtype=torch.bool)
        if self.rendering == 'forward':
            canvas, table = paint_panoptic_masks(masks[None], labels[None], valid, self.criterion.isthing)
        else:
            canvas, table = score_panoptic_masks(masks[None], labels[None], valid, self.criterion.isthing, scores[None], overlap_threshold=0.)
        info = [dict(id=i, category_id=int(c), isthing=bool(self.criterion.isthing[c]))
                for i, c in enumerate(table[0].tolist()) if i and c >= 0]
        result = dict(panoptic_seg=(canvas[0], info))
        if 'panoptic_ground_truth' in item:
            # Oracle evaluation needs the full pool before decoder selection.
            all_masks = F.interpolate(proposal['masks'][None].float(), padded_size,
                                      mode='bilinear', align_corners=False)[0]
            result['panoptic_proposals'] = sem_seg_postprocess(
                all_masks, resized, height, width)
        if self.semantic_on:
            semantic = masks.new_zeros((len(self.criterion.isthing), height, width))
            for segment in info:
                semantic[segment['category_id']] += (canvas[0] == segment['id']).to(semantic)
            result['sem_seg'] = semantic
        if self.instance_on:
            keep = self.criterion.isthing[labels]
            instances = Instances((height, width))
            instances.pred_masks = masks[keep] > 0
            instances.pred_classes = labels[keep]
            instances.scores = scores[keep]
            instances.pred_boxes = BitMasks(instances.pred_masks).get_bounding_boxes()
            result['instances'] = instances
        return result
