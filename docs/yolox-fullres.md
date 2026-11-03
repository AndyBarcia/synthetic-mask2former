# YOLOX full-resolution mask supervision

Run `python train_net.py --config-file configs/synthetic/yolox_fullres.yaml`
with the synthetic renderer and panoptic evaluator installed, and the evaluator
on PYTHONPATH. This reproduces the full-resolution experiment settings:
337 proposals, 25 selection steps, batch 8, 2000 supervised iterations followed
by 200 decoder-only RL iterations. Initialization uses a fresh random seed.
Choose a fresh OUTPUT_DIR to avoid reusing existing experiment artifacts.

GT masks retain their original padded resolution for thing/stuff BCE and Dice,
assignment mask costs, and teacher-target Hungarian matching. Predicted stride-4
logits are bilinearly upsampled with align_corners=False. Losses use all target
pixels; no nearest-neighbor target resizing occurs. The proposal architecture
and inference resolution are unchanged. Oracle evaluation uses the complete
proposal pool before autoregressive selection and is enabled by default.

The implementation was originally developed in ignored output snapshots;
it is now available in the main source tree. Historical checkpoints and their
source snapshots are preserved. Native model/oracle contracts and a single-pixel
supervision/backpropagation regression test cover the change.

Teacher-forced sequences now use the baseline HungarianMatcher with configured
CLASS_WEIGHT, MASK_WEIGHT, DICE_WEIGHT and TRAIN_NUM_POINTS. It computes the
baseline negative class-probability cost and point-sampled mask BCE/Dice costs
against full-resolution GT. YOLOX retains full per-class proposal logits for
this matching; hard argmax labels and IoU thresholds never reject assignments.
The legacy MATCH_IOU_THRESHOLD config is accepted but no longer filters teacher
targets. Only empty visible GT masks are omitted. Dynamic-k proposal supervision
and inference are unchanged. Prefix-tree paths use the baseline shared prefix
and independently permuted suffixes.

## Dynamic-k grouped teacher targets

`configs/synthetic/yolox_dynamic_k_grouped.yaml` opts into
`MODEL.YOLOX.OBJECT_DEC_SUPERVISION: dynamic_k_grouped`; the default remains
`hungarian`. Grouped mode reuses the exact assignment computed for proposal
mask loss, without a second assignment or class/IoU filtering. Each thing GT
owns its dynamic-k mask-positive proposals. The minimum-cost member is the
teacher token; all group members contribute probability to the decoder target:

`loss = -log(sum(softmax(next_token_logits)[group_members]))`.

Compute the sum in log space. Preserve prefix-tree edge/path counts and use
singleton targets for EOF and present stuff classes. This is probability-mass
supervision, not an average of per-proposal cross entropies or a uniform-label
target. It allows concentration within a GT group.

Dynamic-k conflicts are repaired for this mode: give an uncovered GT its
lowest-cost unused candidate, or transfer one from a GT with multiple matches.
Both mask supervision and the teacher groups use the repaired assignment, so
their positive identities remain aligned. Each candidate belongs to only one
GT. Representatives are ranked using the existing assignment cost. If TOPK
drops any mask-positive proposal or a present stuff token, training fails with
an explicit error rather than silently omitting it from decoder supervision;
the all-337 experiment retains every candidate. Empty scenes use EOF alone.

Sequence permutations and inference are unchanged from the Hungarian test.
Inference still excludes exact emitted IDs, not whole GT groups (GT is not
available during inference). Full-resolution targets and oracle evaluation
remain enabled. `teacher/group_size_mean`, `teacher/positive_proposals`, and
`teacher/gt_segments` record training target statistics.

## Concatenated proposal tokens

`configs/synthetic/yolox_dynamic_k_concat.yaml` enables dynamic masks and
`MODEL.YOLOX.TOKEN_FUSION: concat`. The existing residual adapter already receives
the dynamic parameters and spatial context, but compresses them into an additive
mask-dimensional residual through a zero-initialized projection. Concatenation
instead feeds the decoder the original mask embedding followed by all 169 dynamic
MLP parameters, normalized center/stride, mask bias, score and class one-hot.
The decoder's input/output vocabulary projections accept the expanded dimension
(437 for 256 mask features and 6 classes). It is used for teacher feedback,
vocabulary attention and output scoring, including cached inference and RL.
There is no zero-initialized adapter between kernel parameters and the decoder,
so object loss can reach the dynamic branch immediately. The default remains
`residual`, preserving existing checkpoints. Concat changes decoder projection
shapes and should be trained with a fresh decoder. It keeps grouped dynamic-k
supervision, full-resolution targets, and the original best-proposal suffixes.
