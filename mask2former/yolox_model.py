"""Detectron2 meta-architecture for dense mask proposals and an object decoder."""
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.utils.events import get_event_storage

from .modeling.pixel_decoder.fpn import build_pixel_decoder
from .modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from .modeling.yolox.rendering import paint_panoptic_masks, score_panoptic_masks
from .modeling.yolox.proposals import YOLOXMaskProposals, proposal_loss
from .modeling.yolox.mask_first import MaskFirstProposals, mask_first_loss
from .modeling.yolox.policy import DynamicMaskPolicy, ProposalObjectDecoder


class ImageMemories(nn.Module):
    """Project the three pixel-decoder scales into the object decoder's memory."""

    def __init__(self, channels, hidden_dim, enforce_projection=False):
        super().__init__()
        self.input_proj = nn.ModuleList([
            nn.Conv2d(channels, hidden_dim, 1) if channels != hidden_dim or enforce_projection else nn.Sequential()
            for _ in range(3)
        ])
        self.level_embed = nn.Embedding(3, hidden_dim)
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

    def forward(self, multi):
        return [(self.input_proj[i](x).flatten(2) + self.level_embed.weight[i][None, :, None]
                 + self.pe_layer(x, None).flatten(2)).transpose(1, 2) for i, x in enumerate(multi)]


@META_ARCH_REGISTRY.register()
class YOLOXObjectDecoder(nn.Module):
    """Standard loss-dict / batched prediction interface used by ``train_net.py``.

    There is no Mask2Former proposal transformer in this architecture. The
    backbone, pixel decoder, dense proposal head and object policy are registered
    modules, so Detectron2 owns optimization, DDP, evaluation and checkpointing.
    """

    @configurable
    def __init__(self, *, backbone, pixel_decoder, proposal_head, memory, policy,
                 metadata, pixel_mean, pixel_std, freeze_backbone, freeze_pixel_decoder,
                 freeze_memory, freeze_proposals, budget, max_steps, rendering,
                 semantic_on=False, instance_on=False, rl_weight=0., rl_only=False,
                 rl_samples=4, sequence_weight=1., pad_size=0):
        super().__init__()
        if budget <= 0 or max_steps <= 0 or rendering not in ('forward', 'pixel-score'):
            raise ValueError('Positive proposal/action budgets and a supported renderer are required')
        if rl_only and rl_weight <= 0:
            raise ValueError('Decoder-only RL requires a positive RL weight')
        if rl_only and not (freeze_backbone and freeze_pixel_decoder and freeze_memory and freeze_proposals):
            raise ValueError('Decoder-only RL requires all feature and proposal modules frozen')
        self.backbone = backbone
        # Preserve conventional pixel-decoder names for Detectron2 initialization.
        self.sem_seg_head = nn.Module()
        self.sem_seg_head.pixel_decoder = pixel_decoder
        self.sem_seg_head.proposal_head = proposal_head
        self.sem_seg_head.memory = memory
        self.policy = policy
        self.metadata = metadata
        self.freeze_backbone = freeze_backbone
        self.freeze_pixel_decoder = freeze_pixel_decoder
        self.freeze_memory = freeze_memory
        self.freeze_proposals = freeze_proposals
        self.pad_size = pad_size
        self.budget, self.max_steps = budget, max_steps
        self.rendering = rendering
        self.semantic_on, self.instance_on = semantic_on, instance_on
        self.rl_weight, self.rl_only, self.rl_samples = rl_weight, rl_only, rl_samples
        self.sequence_weight = sequence_weight
        for module, frozen in self._freeze_modules():
            module.requires_grad_(not frozen)
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor(pixel_std).view(-1, 1, 1), persistent=False)
        things = set(metadata.thing_dataset_id_to_contiguous_id.values())
        self.register_buffer('isthing', torch.tensor([i in things for i in range(proposal_head.num_classes)]), persistent=False)
        self.train(self.training)

    @classmethod
    def from_config(cls, cfg):
        y, m = cfg.MODEL.YOLOX, cfg.MODEL.MASK_FORMER
        if not y.DYNAMIC_MASKS:
            raise ValueError('YOLOXObjectDecoder requires dynamic-mask tokens')
        if not cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
            raise ValueError('YOLOXObjectDecoder requires panoptic evaluation enabled')
        if cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM != cfg.MODEL.SEM_SEG_HEAD.MASK_DIM:
            raise ValueError('Dense pyramid channels and mask dimension must match')
        if y.WIDTH % 8 or y.MATCH_IOU_THRESHOLD < 0 or y.MATCH_IOU_THRESHOLD > 1:
            raise ValueError('WIDTH must be divisible by 8 and matching IoU must be in [0, 1]')
        if y.RL_SAMPLES < 2:
            raise ValueError('RLOO requires at least two samples')
        backbone = build_backbone(cfg)
        pixel = build_pixel_decoder(cfg, backbone.output_shape())
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        things = metadata.thing_dataset_id_to_contiguous_id.values()
        if not len(things) or len(things) == classes:
            raise ValueError('The dense panoptic head requires thing and stuff classes')
        head_type = MaskFirstProposals if y.BOX_FREE else YOLOXMaskProposals
        options = dict(soft_labels=y.SOFT_LABELS, full_mask_loss=y.FULL_MASK_LOSS, large_kernel=y.LARGE_KERNEL)
        if not y.BOX_FREE:
            options['dynamic_masks'] = True
        head = head_type(cfg.MODEL.SEM_SEG_HEAD.MASK_DIM, classes, things, y.WIDTH, **options)
        decoder = ProposalObjectDecoder(
            cfg.MODEL.SEM_SEG_HEAD.MASK_DIM, m.HIDDEN_DIM, y.TOPK, m.OBJECT_DEC_LAYERS,
            m.NHEADS, m.DIM_FEEDFORWARD, mask_iou_threshold=1.,
            prefix_tree_branches=m.OBJECT_DEC_PREFIX_TREE_BRANCHES,
            position_encoding=m.OBJECT_DEC_POSITION_ENCODING, rope=m.OBJECT_DEC_ROPE,
            eof_threshold=m.OBJECT_DEC_EOF_THRESHOLD, collapse_masks=False)
        rl_only = m.OBJECT_RL_ONLY
        return dict(
            backbone=backbone, pixel_decoder=pixel, proposal_head=head,
            memory=ImageMemories(cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM, m.HIDDEN_DIM, m.ENFORCE_INPUT_PROJ),
            policy=DynamicMaskPolicy(decoder, classes, cfg.MODEL.SEM_SEG_HEAD.MASK_DIM, y.MATCH_IOU_THRESHOLD),
            metadata=metadata, pixel_mean=cfg.MODEL.PIXEL_MEAN, pixel_std=cfg.MODEL.PIXEL_STD,
            freeze_backbone=y.FREEZE_BACKBONE or rl_only,
            freeze_pixel_decoder=y.FREEZE_PIXEL_DECODER or rl_only,
            freeze_memory=y.FREEZE_MEMORY or rl_only, freeze_proposals=y.FREEZE_PROPOSALS or rl_only,
            budget=y.TOPK, max_steps=y.MAX_STEPS, rendering=y.RENDERING,
            semantic_on=m.TEST.SEMANTIC_ON, instance_on=m.TEST.INSTANCE_ON,
            rl_weight=m.OBJECT_RL_WEIGHT, rl_only=rl_only, rl_samples=y.RL_SAMPLES,
            sequence_weight=y.SEQUENCE_LOSS_WEIGHT, pad_size=y.PAD_SIZE)

    def _freeze_modules(self):
        return [(self.backbone, self.freeze_backbone),
                (self.sem_seg_head.pixel_decoder, self.freeze_pixel_decoder),
                (self.sem_seg_head.memory, self.freeze_memory),
                (self.sem_seg_head.proposal_head, self.freeze_proposals)]

    def train(self, mode=True):
        super().train(mode)
        for module, frozen in self._freeze_modules():
            if frozen:
                module.eval()
        return self

    @property
    def device(self):
        return self.pixel_mean.device

    def _targets(self, inputs, images):
        height, width = images.tensor.shape[-2:]
        targets = []
        for item in inputs:
            instances = item['instances'].to(self.device)
            masks = instances.gt_masks
            masks = masks.tensor if isinstance(masks, BitMasks) else masks
            padded = masks.new_zeros((len(masks), height, width))
            padded[:, :masks.shape[-2], :masks.shape[-1]] = masks
            targets.append(dict(labels=instances.gt_classes, masks=padded))
        return targets

    def forward(self, batched_inputs):
        tensors, resized_sizes = [], []
        for item in batched_inputs:
            image = item['image'].to(self.device).float()
            h, w = image.shape[-2:]
            resized_sizes.append(item.get('valid_resized_size', (h, w)))
            if self.pad_size:
                if max(h, w) > self.pad_size:
                    raise ValueError('Input exceeds YOLOX.PAD_SIZE; resize first or set PAD_SIZE=0')
                image = F.pad(image, (0, self.pad_size - w, 0, self.pad_size - h), value=128.)
            tensors.append((image - self.pixel_mean) / self.pixel_std)
        images = ImageList.from_tensors(tensors, self.backbone.size_divisibility)
        # Frozen modules still propagate gradients to their inputs when needed.
        features = self.backbone(images.tensor)
        dense, _, multi = self.sem_seg_head.pixel_decoder.forward_features(features)
        memories = self.sem_seg_head.memory(multi)
        sizes = [tuple(x.shape[-2:]) for x in multi]
        size = tuple(images.tensor.shape[-2:])
        head = self.sem_seg_head.proposal_head
        raw = head(multi, dense, size)
        proposals = [head.proposals(raw, dense, index=i, topk=self.budget) for i in range(len(batched_inputs))]
        if any(len(p['masks']) != self.budget for p in proposals):
            raise ValueError('Proposal budget exceeds the available candidates for this image size')
        if self.training:
            targets = self._targets(batched_inputs, images)
            losses = {}
            if not self.rl_only:
                if not self.freeze_proposals or not self.freeze_pixel_decoder or not self.freeze_backbone:
                    objective = mask_first_loss if head.box_free else proposal_loss
                    losses['loss_proposal'], metrics = objective(head, raw, dense, targets)
                    try:
                        storage = get_event_storage()
                    except AssertionError:
                        storage = None
                    if storage is not None:
                        storage.put_scalars(**{'proposal/' + k: v for k, v in metrics.items()})
                losses['loss_object_decoder'] = self.sequence_weight * self.policy.supervised_loss(
                    proposals, targets, memories, sizes, size)
            if self.rl_weight:
                from .modeling.yolox.rl import policy_rl_loss
                loss, metrics = policy_rl_loss(self.policy, proposals, targets, memories, sizes, size,
                                               self.isthing, self.rl_samples, self.max_steps, self.rendering)
                losses['loss_object_rl'] = self.rl_weight * loss
                try:
                    get_event_storage().put_scalars(**{'rl/' + k: v for k, v in metrics.items()})
                except AssertionError:
                    pass
            return losses
        orders = self.policy.select(proposals, memories, sizes, size, self.max_steps)
        return [self._inference(p, order, item, resized, size)
                for p, order, item, resized in zip(proposals, orders, batched_inputs, resized_sizes)]

    def _inference(self, proposal, order, item, resized, padded_size):
        ids = torch.as_tensor(order, device=self.device, dtype=torch.long)
        masks = proposal['masks'][ids]
        height, width = item.get('height', resized[0]), item.get('width', resized[1])
        if len(ids):
            masks = F.interpolate(masks[None].float(), padded_size, mode='bilinear', align_corners=False)[0]
            masks = masks[:, :resized[0], :resized[1]]
            masks = F.interpolate(masks[None], (height, width), mode='bilinear', align_corners=False)[0]
        else:
            masks = masks.new_empty((0, height, width))
        labels, scores = proposal['labels'][ids], proposal['scores'][ids]
        valid = torch.ones((1, len(ids)), device=self.device, dtype=torch.bool)
        if self.rendering == 'forward':
            canvas, table = paint_panoptic_masks(masks[None], labels[None], valid, self.isthing)
        else:
            canvas, table = score_panoptic_masks(masks[None], labels[None], valid, self.isthing, scores[None], overlap_threshold=0.)
        info = [dict(id=i, category_id=int(c), isthing=bool(self.isthing[c]))
                for i, c in enumerate(table[0].tolist()) if i and c >= 0]
        result = dict(panoptic_seg=(canvas[0], info))
        if self.semantic_on:
            semantic = masks.new_zeros((len(self.isthing), height, width))
            for segment in info:
                semantic[segment['category_id']] += (canvas[0] == segment['id']).to(semantic)
            result['sem_seg'] = semantic
        if self.instance_on:
            keep = self.isthing[labels]
            instances = Instances((height, width))
            instances.pred_masks = masks[keep] > 0
            instances.pred_classes = labels[keep]
            instances.scores = scores[keep]
            instances.pred_boxes = BitMasks(instances.pred_masks).get_bounding_boxes()
            result['instances'] = instances
        return result
