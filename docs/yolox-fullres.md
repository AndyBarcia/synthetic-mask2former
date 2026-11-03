# Default synthetic YOLOX recipe

Run `python train_net.py --config-file configs/synthetic/yolox.yaml` with the
synthetic renderer and panoptic evaluator available. Use a fresh OUTPUT_DIR.
The recipe matches output/small_synthetic_yolox_all_masks_fullres_dynamic_k_rl:
337 proposals, 25 selection steps, batch 8, FP32, 2000 supervised iterations
followed by 200 decoder-only RL iterations, and 100 validation scenes. Model
initialization is random; renderer seeds and evaluation batches are fixed.

YOLOX always uses dynamic masks and the original zero-initialized residual
proposal-token adapter. It preserves full-resolution GT for dense thing/stuff
BCE and Dice, dynamic-k assignment costs, and PQ rewards. Predicted logits are
bilinearly upsampled with align_corners=False. GT masks are never downsampled.

The decoder reuses exactly the dynamic-k assignment that supervises proposal
masks. Conflicts are repaired to cover each GT with disjoint positive groups.
Each teacher token is the minimum-cost member of its GT group. Shared prefixes
and permuted suffixes keep these same representatives; suffix proposal sampling
is absent. The loss is `-log(sum(softmax(logits)[group_members]))`, preserving
prefix-tree path/edge counts. Stuff and EOF are singleton targets. This is
probability-mass supervision, not uniform-label cross entropy.

TOPK must retain all mask-positive proposals and present stuff tokens. The
128x128 recipe retains every candidate. Missing positives raise an error.
Inference excludes exact emitted IDs and paints later masks over earlier ones.
Synthetic oracle evaluation is enabled by default and sees the full proposal
pool before selection. `teacher/group_size_mean`, `teacher/positive_proposals`,
and `teacher/gt_segments` report training target statistics.

Hungarian teacher matching, concatenated tokens, linear-only masks, alternate
YOLOX rendering, and their experiment switches/configs are removed. Old research
configs are not accepted by the current schema. Saved output snapshots and
checkpoints remain reference artifacts, not active configuration sources.
