# Multiple whole-object proposals

The default experiment config is
`configs/synthetic-scene/panoptic-segmentation/maskseq_R50_bs16_10kpre_2krl_multiple_proposals.yaml`.
It uses three Hungarian slots per GT, retaining the highest-IoU match
unconditionally. The second and third IoU-ranked matches require IoU strictly
above 0.5 and 0.9, respectively, and the correct predicted class. Each auxiliary
decoder layer applies the same rule to its own predictions. This is the default
for `tools/smoke_multiple_proposals.py`; the named k2 config remains available
for unfiltered baseline experiments. The schedule is 10k supervised followed by
2k decoder-only RL through the existing phase workflow.

`MODEL.MASK_FORMER.PROPOSALS_PER_GT` defaults to 1. Set it to 2 to repeat every
whole-object GT twice for Hungarian matching while retaining the configured query
count (100 in the experiment config). Every matched query learns the same parent
class and complete visible mask. Unmatched queries learn no-object. Each mask
decoder layer matches independently and uses the existing configurable point or
block mask/Dice losses, normalized by the distributed matched count. No part
annotations, fused targets, object-group separators, or union losses are involved.

Matching slots map back to original object IDs for teacher construction. Each
candidate samples one query uniformly from each represented object's alternatives
and shuffles object order. The highest whole-scene PQ candidate is teacher-forced
with one query per object followed by EOF. Set `OBJECT_PRETRAIN_BEST_OF_K` to
control the candidate count (32 by default); setting it to 0 retains legacy random
prefix-tree teachers when `PROPOSALS_PER_GT=1`. Inference and decoder-only RL use
the original whole-object decoder and unrepeated GT. No proposal-diversity penalty
is added.

`STANDARD_CLASSIFICATION` enables ordinary class-plus-no-object cross entropy.
Matching uses class probabilities normalized including no-object; the separate
query-bias head is frozen and its loss/matching cost disabled. Foreground confidence
replaces query bias at inference. Set it to false to retain the legacy foreground
classification/query-bias path. Multiple-proposal mode requires it enabled, no fused
targets, and best-of-K teachers. Hybrid selected-panoptic training is excluded.

If repeated GTs exceed the query count, rectangular Hungarian matching keeps the
best query-count matches. Metrics `multiple_proposals/objects_without_match` and
`queries_per_represented_object` expose omitted objects and achieved multiplicity.

The config
`configs/synthetic-scene/panoptic-segmentation/maskseq_R50_bs16_10kpre_2krl_multiple_proposals_k2.yaml`
uses K=2, 100 queries, global batch 16, seed42, fresh ImageNet R50, and the 10k
supervised/2k decoder-only batch-RQ RL schedule. The global batch remains 16 on
two GPUs; learning rates do not scale with GPU count. Use the existing phase-config
workflow to switch from supervised to RL and load the supervised checkpoint.
Always launch long training in a detached persistent process and record logs/status.

The research run at `output/synth_pan_R50_bs16_10kpre_2krl_multiple_proposals_k2`
used full-resolution CUDA losses in its pinned source snapshot. This commit
intentionally excludes that independent backend; its portable config uses point
sampling, so it preserves the proposal/teacher scheme but does not exactly
reproduce that run's mask-loss objective.

Validation: `python -m unittest discover -s tests -p test_multiple_proposals.py -v`
checks alternative sampling, sample-major best-teacher selection, repeated targets,
no-object gradients, matching probabilities, overflow and auxiliary loss gradients
with both point and block losses. `python tools/smoke_multiple_proposals.py` runs
GPU supervised forward/backward, inference and frozen-proposal RL without the
full-resolution backend. `tools/inspect_duplicate_proposals.py` measures paired
mask similarity for a run directory with pinned source and phase config.

## Dynamic mask assignment

Set `PROPOSAL_ASSIGNMENT: dynamic_k` and `PROPOSALS_PER_GT: 1` to use original
whole-object targets with variable multiplicity. `DYNAMIC_K_TOPK` defaults to 10.
For each GT, sum the top 10 binary mask IoUs across queries, floor the sum, and
clamp to at least one. Select that many lowest-cost queries using the existing
weighted classification/BCE/Dice costs. A query claimed by multiple GTs goes to
its globally lowest-cost GT, following YOLOX SimOTA conflict resolution. This can
leave a GT without a match; the existing objects-without-match metric reports it.
There is no box-center prior because these are unanchored mask queries.

IoUs use bilinear-upsampled mask logits thresholded at zero at padded GT
resolution, calculated in float32 even under AMP; empty/empty IoU is zero.
Each auxiliary decoder layer assigns independently and normalizes its mask loss
by its distributed matched count. Teacher candidates sample one query from each
represented original object. All existing multiple-proposal compatibility
restrictions apply, and decoder-only RL retains the frozen mask proposals.
This changes assignment multiplicity, without a proposal-diversity loss.

Reference: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
Tests: `python -m unittest discover -s tests -p test_dynamic_k_assignment.py -v`.

## Filtered fixed-k Hungarian assignment

Use `PROPOSAL_ASSIGNMENT: hungarian`, `PROPOSALS_PER_GT: 3`, and
`EXTRA_PROPOSAL_MIN_IOU: 0.5` to filter the resulting three matches per object.
The highest-IoU match for every represented GT is always retained, even below
threshold. Additional matches survive only at IoU strictly greater than 0.5;
rejected queries receive the usual no-object classification target and no mask
loss. Filtering does not rerun Hungarian or reallocate rejected matches.

IoU uses binary, bilinear-upsampled masks at padded GT resolution in float32,
as in the dynamic-k experiment. Every decoder layer filters its own matches and
normalizes by its retained distributed match count. Decoder teachers sample
only retained final-layer matches, with one query per represented GT.
`multiple_proposals/filtered_queries` logs removals per image; existing metrics
track coverage and retained multiplicity. Filtering defaults to disabled (-1).
It preserves coverage of the original Hungarian assignment; query overflow
can still omit GTs as in unfiltered fixed-k training.

Tests: `python -m unittest discover -s tests -p test_filtered_proposals.py -v`.

Set `EXTRA_PROPOSAL_REQUIRE_CORRECT_CLASS: true` to additionally require that
each extra query's argmax class (including no-object) equals the GT class.
The highest-IoU base match remains unconditional to preserve coverage. Each
auxiliary layer uses its own class logits, and the default is false.

For tiered filtering, set `EXTRA_PROPOSAL_IOU_THRESHOLDS: [0.5, 0.9]` with
`PROPOSALS_PER_GT: 3` and filtering enabled. Within each object's Hungarian
matches, rank by descending binary mask IoU. Always retain rank 1; rank 2
requires IoU > 0.5 and rank 3 requires IoU > 0.9. These per-rank thresholds
override the scalar threshold, and the optional correct-class gate still
applies to both extras. Query index and repeated-target slot do not define rank.
Threshold lists must be nondecreasing and have one entry per extra proposal.
