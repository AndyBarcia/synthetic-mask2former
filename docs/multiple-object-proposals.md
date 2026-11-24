# Multiple whole-object proposals

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
