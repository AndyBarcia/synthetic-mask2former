# Dense mask proposals with the object decoder

`YOLOXObjectDecoder` is a Detectron2 meta-architecture. Use the existing
`train_net.py` entry point: the regular trainer owns distributed training,
optimizer, LR scheduler, checkpointing, resume, and evaluators. The production
model never imports experiment code from `tools/` or saved source snapshots.

## Architecture

The backbone and pixel decoder produce a stride-4 mask basis and three pyramid
features. A decoupled convolutional proposal head predicts class/objectness,
linear mask kernels, and 169 dynamic residual parameters per spatial location.
Thing masks use an eight-channel dynamic feature map and relative coordinates;
stuff uses class-specific context-conditioned kernels. The object policy combines
mask-kernel embeddings with a zero-initialized residual adapter over dynamic
parameters, coordinates, score, bias, and class. Image memories use their own
three-scale projections, position encodings, and level embeddings.

Global score-only top-300 selection precedes mask rendering. There is no
Mask2Former proposal transformer, NMS, hard overlap exclusion, or box cropping.
The default dynamic variant uses boxes for training assignment and regression;
`BOX_FREE: true` selects mask-first assignment and removes the box heads/loss.
The object decoder excludes only exact previously emitted proposal IDs. The
64-action cap includes a possible EOF action, matching the experiments.

Training returns `loss_proposal` and `loss_object_decoder`, using dynamic-k
matching, thing/stuff mask losses and four shuffled class-aware mask-matched
prefix-tree target paths. Discrete selection/matching/regions are not
backpropagated; selected tokens and image memories retain feature gradients.

Inference emits Detectron2 `panoptic_seg` outputs. Optional semantic maps and
thing `Instances` are also supported. Forward painting is the default: later
selected masks overwrite earlier ones and same-class stuff is merged.
`MODEL.YOLOX.RENDERING: pixel-score` enables score-weighted pixel competition.
Both renderers use zero retained-area rejection. The same renderer is used by RL.

## Train without COCO-pretrained segmentation weights

```bash
python train_net.py --num-gpus 2 \
  --config-file configs/coco/panoptic-segmentation/yolox_R50_bs16_100k.yaml
```

This configuration initializes **only the backbone** from ImageNet. The pixel
decoder, proposal head, memory projections and object decoder start fresh, and
all are trainable. The box-free alternative is
`yolox_R50_bs16_100k_mask_first.yaml`.

The recipe uses 512-square LSJ training, batch 16, seed 42, AdamW, no warmup,
100k updates, 10x LR drops at 80k/95k, checkpoints and full validation every 5k.
Policy LR is 1e-4; feature/proposal LR is 1e-5 via
`MODEL.YOLOX.FEATURE_LR_MULTIPLIER`. Backbone LR uses the existing
`SOLVER.BACKBONE_MULTIPLIER`. Gradient clipping is independently configured for
head (10), policy (1), and feature modules (1), matching the pilot's head/pixel/
policy limits. AMP is disabled in the example recipe, as in the pilots.

Launch long training in a detached persistent tmux session or service, preserve
its console log, and record its exit status. `--resume` restores the native
model, optimizer, scheduler, and iteration via Detectron2's `last_checkpoint`.

## Freeze settings and validation geometry

`MODEL.YOLOX.FREEZE_BACKBONE`, `FREEZE_PIXEL_DECODER`, `FREEZE_MEMORY`, and
`FREEZE_PROPOSALS` independently control optimization and evaluation mode.
Frozen modules still propagate gradients to trainable inputs where needed.
The `pixel_finetune` config freezes the backbone and memory projections, matching
the current experiment. Set `MODEL.WEIGHTS` explicitly when using that config;
it is a fine-tuning template, not a randomly initialized frozen-feature recipe.

`PAD_SIZE: 512` enables deterministic validation that resizes the long edge to
512, pads raw pixels with 128, and records the unpadded size. The model crops
padding before resizing to the original resolution. This matches the research
validation geometry. The native test mapper reads no GT annotations. Set
`PAD_SIZE: 0` to use the standard Detectron2 test mapper and variable-size
inference instead; scores under that geometry are a different comparison.

## Import an existing experiment

Research `joint_*.pt` files are not Detectron2 checkpoints. Convert them once:

```bash
python tools/convert_yolox_checkpoint.py \
  --config-file configs/coco/panoptic-segmentation/yolox_R50_bs16_100k_pixel_finetune.yaml \
  --base-checkpoint output/coco_pan_R50_official_joint_mask_object_supervised_100k_cont10k/model_final.pth \
  --joint-checkpoint output/joint_dynamic_policy_pixel_100k/joint_030000.pt \
  --output output/yolox_native_030000.pth

python train_net.py --eval-only --num-gpus 2 \
  --config-file configs/coco/panoptic-segmentation/yolox_R50_bs16_100k_pixel_finetune.yaml \
  MODEL.WEIGHTS output/yolox_native_030000.pth
```

The converter validates every tensor and transfers the frozen backbone, trained
pixel decoder (when present in the joint file), proposal head, policy/adapter,
and the original memory projection/level weights. It refuses missing tensors,
shape mismatches, budget mismatches, or an existing output file. Use
`--head-checkpoint` instead of `--joint-checkpoint` to reproduce the original
joint-training initialization from the base object decoder and a proposal head.
Choose a config matching the source width, decoder depth, and box-free variant.
The converter starts a fresh optimizer/schedule; it does not claim an exact
resume of a research training run. Subsequent native checkpoints are complete
and require no reference to the original base checkpoint.

## Decoder-only RL

`yolox_R50_decoder_rl_2k.yaml` uses the standard training phase infrastructure
with four-sample RLOO plus the greedy baseline and per-image training-canvas PQ.
Pass a **native supervised checkpoint** as `MODEL.WEIGHTS`. Backbone, pixel
decoder, memory projection and proposal head freeze automatically for pure RL;
the object decoder and adapter train. Capped sampled traces are replayed exactly,
without appending an artificial EOF action. Install the existing
`panoptic-evaluator` dependency or add its checkout to `PYTHONPATH` for RL.
Hybrid phases combine the proposal/sequence losses with the same RL objective.
Other experimental Mask2Former-specific reward choices and query refiners do
not apply to this architecture.

## Validation

Run `python -m unittest discover -s tests -p test_native_yolox.py` in the project
container. Tests cover both proposal variants, empty GT, pixel gradients and
backbone freezing, no overlap exclusions, pixel/forward prediction contracts,
validation padding, checkpoint reload, decoder RL, and exact research-head
output parity. Integration evidence is recorded under
`output/yolox_native_integration/` (native two-GPU train/eval/resume and converted
30k full validation).

The migrated 30k checkpoint was evaluated on all 5,000 COCO validation images
using native `train_net.py --eval-only --num-gpus 2`: **PQ 38.02091539113977,
SQ 78.22189495075814, RQ 46.86017877905489**. All nine overall/thing/stuff
PQ/SQ/RQ fields match the research evaluation exactly. The two-GPU native
trainer smoke also saved checkpoints, evaluated, resumed the optimizer and
scheduler at iteration 2, and completed a third update with the backbone frozen
and pixel decoder updating on both ranks.
