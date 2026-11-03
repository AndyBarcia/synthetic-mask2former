# YOLOX proposals with the MaskFormer training structure

The YOLOX path uses the same boundaries as the Mask2Former implementation:

| Mask2Former | YOLOX variant | Responsibility |
| --- | --- | --- |
| `MaskFormer` | `YOLOXObjectDecoder(MaskFormer)` | Image normalization/padding, targets, loss weighting, inference |
| `MaskFormerHead` | `YOLOXHead` | Pixel decoder, proposal predictions and image/mask embeddings |
| `HungarianMatcher` | `YOLOXMatcher` | Proposal assignment and object-sequence matching |
| `SetCriterion` | `YOLOXCriterion` | Named supervised/RL losses, weights, ownership of the object decoder |

`YOLOXHead` is registered with `SEM_SEG_HEADS_REGISTRY` and built with
`build_sem_seg_head`. Its `predictor` is the dense proposal network instead of
a Mask2Former transformer. It returns `pred_masks`, `mask_embeddings`, object
image memories, and the dense predictions needed for proposal supervision.

`YOLOXCriterion(outputs, targets)` returns unweighted losses. As in MaskFormer,
the meta-architecture applies `criterion.weight_dict`. Losses are separately
logged: `loss_objectness`, `loss_ce`, `loss_box` (unless box-free), `loss_mask`,
`loss_dice`, `loss_stuff_presence`, `loss_stuff_mask`, `loss_stuff_dice`,
`loss_object_decoder`, and optionally `loss_object_rl`. The object decoder lives
under `criterion.object_decoder`, not in a separate policy wrapper.

## Training and configuration

```bash
python train_net.py --num-gpus 2 \
  --config-file configs/coco/panoptic-segmentation/yolox_R50_bs16_100k.yaml
```

The default recipe uses an ImageNet-initialized ResNet-50 backbone and fresh
pixel/proposal/object decoders; all are trainable. It shares the ordinary COCO
training and test loaders, normalization, size-divisible `ImageList` padding,
optimizer, gradient clipping, LR scheduler, checkpoints and evaluators with
Mask2Former. There is no custom YOLOX trainer or test mapper.

The 100k recipe uses batch16, 512 LSJ training, LR drops at80k/95k and 5k
checkpoint/evaluation intervals. `SOLVER.MASK_PROPOSAL_MULTIPLIER` controls the
semantic-head LR relative to the base LR, and `BACKBONE_MULTIPLIER` controls the
backbone LR. Gradient clipping uses the same `SOLVER.CLIP_GRADIENTS` code path
as Mask2Former. Classification and mask-loss weights reuse
`MODEL.MASK_FORMER.CLASS_WEIGHT`, `MASK_WEIGHT` and `DICE_WEIGHT`; the YOLOX
objectness, box and stuff-presence weights live in `MODEL.YOLOX`.

Use `yolox_R50_bs16_100k_mask_first.yaml` for box-free mask assignment and no
box loss. `yolox_R50_bs16_100k_pixel_finetune.yaml` freezes the backbone and
memory projections while training the pixel decoder, predictor, adapter and
object decoder; supply appropriate `MODEL.WEIGHTS` for that fine-tuning template.
The `FREEZE_BACKBONE`, `FREEZE_PIXEL_DECODER`, `FREEZE_MEMORY` and
`FREEZE_PROPOSALS` flags control gradient updates and evaluation mode.

Use `--resume` for native checkpoint/optimizer/scheduler/iteration restoration.
Launch long training in a persistent detached tmux session or service and retain
its console log and exit status.

## Proposals and inference

The dense head produces class/objectness scores, linear mask kernels and compact
dynamic residual masks on three pyramid levels. Score-only top-K selection
(default300) precedes rendering. The default variant uses boxes for assignment
and regression; `BOX_FREE: true` uses mask-center and mask-quality costs instead.
Neither variant uses box NMS, mask NMS, hard overlap exclusion, or box cropping.
The object decoder excludes only previously selected IDs and learns selection.

`MODEL.YOLOX.RENDERING: forward` is the default: later selected masks overwrite
earlier ones and same-class stuff is merged. `pixel-score` enables score-weighted
pixel competition. Outputs follow the standard Detectron2 `panoptic_seg` format,
with optional semantic maps and thing instances.

Evaluation follows the normal Mask2Former test resize and padding settings:
`INPUT.MIN_SIZE_TEST`, `INPUT.MAX_SIZE_TEST`, and model size divisibility. The
old experiment's fixed512 raw-128 padding and `valid_resized_size` convention
have been removed. Consequently old experimental metrics are not an exact parity
target under this implementation. Set identical input settings when comparing
YOLOX and Mask2Former models.

## Existing experiment weights

The optional converter imports the original research head/joint weights:

```bash
python tools/convert_yolox_checkpoint.py \
  --config-file configs/coco/panoptic-segmentation/yolox_R50_bs16_100k_pixel_finetune.yaml \
  --base-checkpoint output/coco_pan_R50_official_joint_mask_object_supervised_100k_cont10k/model_final.pth \
  --joint-checkpoint output/joint_dynamic_policy_pixel_100k/joint_030000.pt \
  --output output/yolox_native_030000.pth
```

It checks shapes and category mappings, imports trained pixel-decoder weights
when present, and writes the current native module layout. It initializes a new
optimizer/schedule. Intermediate native checkpoints from before the
head/criterion refactor use a different layout; reconvert from the original
research checkpoint. This is a weight importer, not an exact reproduction of
the experiment scaffolding or its evaluation protocol.

## RL

`yolox_R50_decoder_rl_2k.yaml` uses the existing supervised/RL phase machinery.
Pass a native supervised checkpoint as `MODEL.WEIGHTS`. Pure RL freezes the
backbone and semantic-head features/proposals and trains the object decoder and
token adapter. The criterion computes per-image PQ rewards and four-sample RLOO
with the greedy baseline, using the selected inference renderer. Capped traces
are replayed without inventing an EOF action. Hybrid mode combines these losses
with supervised proposal and sequence losses. RL requires the existing
`panoptic-evaluator` dependency on `PYTHONPATH` or installed in the environment.

## Checks

`python -m unittest discover -s tests -p test_native_yolox.py` checks criterion
loss names, both assignment variants, empty targets, feature gradients, freezing,
Detectron2 checkpoint reload, predictions and decoder-only RL. These check model
contracts rather than preserving dummy-experiment outputs.
