# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.object_decoder import ObjectDecoder
from .modeling.deferred_masks import DeferredMaskProposals, render_selected_masks
from .modeling.panoptic_paint import paint_panoptic_masks


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        panoptic_paint_order: str,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        object_rl_only: bool = False,
        inference_mask_optimization: bool = True,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            object_rl_only: bool, freeze mask proposals and train only the object decoder
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.object_rl_only = object_rl_only
        self.inference_mask_optimization = inference_mask_optimization
        if object_rl_only:
            if criterion.object_decoder is None or criterion.object_rl_weight <= 0:
                raise ValueError("RL requires an object decoder and positive RL weight")
            self.backbone.requires_grad_(False)
            self.sem_seg_head.requires_grad_(False)
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        if panoptic_paint_order not in ("reverse", "forward"):
            raise ValueError(f"Unknown panoptic paint order: {panoptic_paint_order}")
        self.panoptic_paint_order = panoptic_paint_order
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        query_bias_weight = cfg.MODEL.MASK_FORMER.QUERY_BIAS_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            cost_query_bias=cfg.MODEL.MASK_FORMER.QUERY_BIAS_MATCHER_WEIGHT,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            mask_loss_type=cfg.MODEL.MASK_FORMER.MASK_LOSS_TYPE,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_query_bias": query_bias_weight,
        }

        use_object_decoder = (
            cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME == "MultiScaleMaskedTransformerDecoder"
        )
        if use_object_decoder:
            weight_dict["loss_object_decoder"] = 1.0
            if cfg.MODEL.MASK_FORMER.OBJECT_RL_WEIGHT > 0:
                weight_dict["loss_object_rl"] = cfg.MODEL.MASK_FORMER.OBJECT_RL_WEIGHT

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items() if k not in ("loss_object_decoder", "loss_object_rl")})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "query_bias"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            mask_loss_type=cfg.MODEL.MASK_FORMER.MASK_LOSS_TYPE,
            fused_thing_masks=cfg.MODEL.MASK_FORMER.FUSED_THING_MASKS,
            thing_class_ids=getattr(
                MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                "thing_dataset_id_to_contiguous_id", {},
            ).values(),
            object_decoder=ObjectDecoder(
                cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
                cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
                cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                cfg.MODEL.MASK_FORMER.OBJECT_DEC_LAYERS,
                cfg.MODEL.MASK_FORMER.NHEADS,
                cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
                cfg.MODEL.MASK_FORMER.OBJECT_DEC_MASK_IOU_THRESHOLD,
                cfg.MODEL.MASK_FORMER.OBJECT_DEC_PREFIX_TREE_BRANCHES,
                position_encoding=cfg.MODEL.MASK_FORMER.OBJECT_DEC_POSITION_ENCODING,
                rope=cfg.MODEL.MASK_FORMER.OBJECT_DEC_ROPE,
            ) if use_object_decoder else None,
            object_rl_weight=cfg.MODEL.MASK_FORMER.OBJECT_RL_WEIGHT,
            object_rl_max_steps=cfg.MODEL.MASK_FORMER.OBJECT_RL_MAX_STEPS,
            object_rl_reward_size=cfg.MODEL.MASK_FORMER.OBJECT_RL_REWARD_SIZE,
            object_rl_num_samples=cfg.MODEL.MASK_FORMER.OBJECT_RL_NUM_SAMPLES,
            object_rl_baseline=cfg.MODEL.MASK_FORMER.OBJECT_RL_BASELINE,
            object_rl_objective=cfg.MODEL.MASK_FORMER.OBJECT_RL_OBJECTIVE,
            object_rl_train_eof=cfg.MODEL.MASK_FORMER.OBJECT_RL_TRAIN_EOF,
        )
        criterion.object_rl_paint_order = cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_PAINT_ORDER

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "panoptic_paint_order": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_PAINT_ORDER,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "object_rl_only": cfg.MODEL.MASK_FORMER.OBJECT_RL_ONLY,
            "inference_mask_optimization": cfg.MODEL.MASK_FORMER.INFERENCE_MASK_OPTIMIZATION,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        predictor = self.sem_seg_head.predictor
        supports_optimization = hasattr(predictor, "inference_mask_optimization")
        efficient = (not self.training and self.inference_mask_optimization
                     and supports_optimization and predictor.inference_mask_optimization)
        deferred = efficient and self.criterion.object_decoder is not None and not self.instance_on
        head_options = (dict(optimize_inference=efficient, defer_mask_predictions=deferred)
                        if supports_optimization else {})
        if self.training and self.object_rl_only:
            self.backbone.eval()
            self.sem_seg_head.eval()
            with torch.no_grad():
                features = self.backbone(images.tensor)
                outputs = self.sem_seg_head(features, **head_options)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features, **head_options)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            if targets is not None and not self.object_rl_only:
                gt_query_bias_logits = [
                    self.sem_seg_head.predictor.query_bias_embed(features, target["masks"])
                    for features, target in zip(outputs["query_bias_features"], targets)
                ]
                outputs["gt_query_bias_logits"] = gt_query_bias_logits
                for aux in outputs.get("aux_outputs", []):
                    aux["gt_query_bias_logits"] = gt_query_bias_logits

            # bipartite matching-based loss
            losses = (self.criterion.loss_object_rl(outputs, targets)
                      if self.object_rl_only else self.criterion(outputs, targets))

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            query_bias_results = outputs["query_bias_logits"].sigmoid()
            generated_objects = (
                self.criterion.object_decoder.generate(
                    outputs["mask_embeddings"], outputs["object_decoder_image_features"],
                    outputs["pred_masks"], outputs["object_decoder_image_sizes"],
                    **({"image_regions": outputs["object_decoder_image_regions"]} if deferred else {}),
                )
                if self.criterion.object_decoder is not None
                else [(None, None, None)] * len(batched_inputs)
            )
            padded_size = images.tensor.shape[-2:]
            if deferred:
                mask_embeddings, mask_features = outputs["mask_embeddings"], outputs["mask_features"]
                mask_pred_results = render_selected_masks(
                    mask_embeddings, mask_features, [order for order, _, _ in generated_objects], padded_size)
            else:
                mask_pred_results = F.interpolate(mask_pred_results, size=padded_size,
                                                  mode="bilinear", align_corners=False)
            del outputs

            processed_results = []
            panoptic_inputs = []
            # Analysis tools can still override the per-image painter.
            batch_paint = (self.panoptic_on and
                           getattr(self.panoptic_inference, "__func__", None) is MaskFormer.panoptic_inference)
            for image_index, (mask_cls_result, mask_pred_result, query_bias_result, generated, input_per_image, image_size) in enumerate(zip(
                mask_cls_results,
                mask_pred_results,
                query_bias_results,
                generated_objects,
                batched_inputs,
                images.image_sizes,
            )):
                object_order, object_log_odds, object_probabilities = generated
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                if object_order is not None:
                    processed_results[-1]["object_decoder"] = {
                        "query_indices": object_order,
                        "mask_vs_eof_logits": object_log_odds,
                        "mask_vs_eof_probabilities": object_probabilities,
                    }

                if deferred:
                    mask_pred_result = mask_pred_result[:len(object_order)]
                    mask_cls_result = mask_cls_result[object_order]
                    query_bias_result = query_bias_result[object_order]
                    # The rendered masks follow the selected order; keep original
                    # query IDs in object_decoder metadata, use local IDs to paint.
                    object_order = torch.arange(len(object_order), device=mask_pred_result.device)

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = (retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    ) if mask_pred_result.shape[0] else mask_pred_result.new_empty((0, height, width)))
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                    query_bias_result = query_bias_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    semantic_query_weights = query_bias_result
                    if object_order is not None:
                        semantic_query_weights = torch.zeros_like(query_bias_result)
                        semantic_query_weights[object_order] = 1
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result, semantic_query_weights
                    )
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    if batch_paint:
                        panoptic_inputs.append((mask_cls_result, mask_pred_result, query_bias_result, object_order))
                    else:
                        processed_results[-1]["panoptic_seg"] = retry_if_cuda_oom(self.panoptic_inference)(
                            mask_cls_result, mask_pred_result, query_bias_result, object_order)
                    if "panoptic_ground_truth" in input_per_image:
                        # Keep every query, before object-decoder/score selection,
                        # for the synthetic evaluator's oracle assignment.
                        processed_results[-1]["panoptic_proposals"] = (
                            DeferredMaskProposals(mask_embeddings[image_index], mask_features[image_index],
                                                  padded_size, image_size, (height, width),
                                                  self.sem_seg_postprocess_before_inference)
                            if deferred else mask_pred_result)
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result, query_bias_result,
                        object_order, object_probabilities
                    )
                    processed_results[-1]["instances"] = instance_r

            if batch_paint:
                # Batch uniform canvases; mixed output dimensions form small groups.
                groups = {}
                for index, item in enumerate(panoptic_inputs):
                    groups.setdefault(tuple(item[1].shape[-2:]), []).append(index)
                for indices in groups.values():
                    painted = self.panoptic_batch_inference([panoptic_inputs[index] for index in indices],
                                                            ordered=deferred)
                    for index, (panoptic, classes) in zip(indices, painted):
                        processed_results[index]["panoptic_seg"] = panoptic
                        processed_results[index]["panoptic_classes"] = classes
            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred, query_bias):
        mask_cls = F.softmax(mask_cls[..., :-1], dim=-1) * query_bias[:, None]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_batch_inference(self, inputs, *, ordered=False):
        masks, labels = [], []
        for mask_cls, mask_pred, query_bias, object_order in inputs:
            if object_order is None:
                class_scores, categories = F.softmax(mask_cls[..., :-1], dim=-1).max(-1)
                scores = class_scores * query_bias
                kept = torch.where(scores > self.object_mask_threshold)[0]
                object_order = kept[torch.argsort(scores[kept], descending=True)]
            else:
                categories = mask_cls[..., :-1].argmax(-1)
            masks.append(mask_pred if ordered else mask_pred[object_order])
            labels.append(categories if ordered else categories[object_order])
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=-torch.inf)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(row) for row in labels], device=masks.device)
        valid = torch.arange(masks.shape[1], device=masks.device)[None] < lengths[:, None]
        thing_ids = set(self.metadata.thing_dataset_id_to_contiguous_id.values())
        isthing = torch.tensor([category in thing_ids for category in range(inputs[0][0].shape[-1] - 1)],
                               device=masks.device, dtype=torch.bool)
        maps, classes = paint_panoptic_masks(masks, labels_padded, valid, isthing,
                                            reverse=self.panoptic_paint_order == "reverse")
        # One small transfer for the legacy segments_info dictionaries; the
        # evaluator consumes the class tensors directly without uploading them.
        class_rows = classes.tolist()
        return [((maps[image], [{"id": slot, "category_id": category, "isthing": category in thing_ids}
                               for slot, category in enumerate(row) if slot and category >= 0]), classes[image])
                for image, row in enumerate(class_rows)]

    def panoptic_inference(self, mask_cls, mask_pred, query_bias, object_order=None):
        return self.panoptic_batch_inference([(mask_cls, mask_pred, query_bias, object_order)])[0][0]

    def instance_inference(self, mask_cls, mask_pred, query_bias,
                           object_order=None, object_probabilities=None):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        class_scores, labels_per_image = F.softmax(mask_cls[:, :-1], dim=-1).max(-1)
        order = (torch.argsort(query_bias, descending=True, stable=True)
                 if object_order is None else object_order)
        scores_per_image = (class_scores[order] * query_bias[order]
                            if object_order is None else object_probabilities.to(mask_pred))
        labels_per_image = labels_per_image[order]
        mask_pred = mask_pred[order]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        if object_order is None:
            mask_scores = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1)
            mask_scores /= result.pred_masks.flatten(1).sum(1) + 1e-6
            result.scores = scores_per_image * mask_scores
        else:
            # P(mask) / (P(mask) + P(EOF)) from the generation step.
            result.scores = scores_per_image
        result.pred_classes = labels_per_image
        return result
