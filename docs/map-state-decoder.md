# Two-stage map-state object decoder

`MODEL.MASK_FORMER.OBJECT_DEC_MAP_ENCODER=True` is the default. Set False to
use the previous autoregressive query-history decoder. The default hidden
width is256; the COCO R50 configuration uses5 pointer layers.

At each prefix, paint selected proposals last-mask-wins on the coarsest image
feature grid. Each surviving query contributes its embedding, image features
averaged over visible pixels, visible/raw area and relative paint rank.
Add an uncovered-region token and learned summary token. Two transformer
encoder layers mix these tokens; the summary feeds the existing pointer
layers with multi-scale image attention and mask-vocabulary attention.
The pointer's image attention excludes the union of selected masks. Segment
image evidence is average-pooled; there is no per-segment image cross-attention.

Prefix nodes are independent map states; they cannot attend to sibling or
future nodes. Fully overwritten regions are masked out. Gradients flow into
query and pooled image features, while binary ownership is discrete.
Checkpointed chunks bound memory; generation encodes only its current map.
The 256d two-stage map checkpoints from the experiments retain compatible
parameter names. Legacy history checkpoints require MAP_ENCODER=False.

## Greedy EOF stopping

`MODEL.MASK_FORMER.OBJECT_DEC_EOF_THRESHOLD=0.5` applies strict EOF probability
>50% after vocabulary exclusions to inference and greedy RL references (RLOO
and best-of-N). Otherwise select the highest-probability mask. Zero restores
legacy individual-token argmax. Sampled RL rollouts retain categorical EOF
sampling and its original likelihood, so EOF keeps its policy-gradient signal.

## Probability-grouped mask selection

`OBJECT_DEC_COLLAPSE_MASKS=True` and `OBJECT_DEC_COLLAPSE_IOU_THRESHOLD=0.8`
are the defaults under `MODEL.MASK_FORMER`. Set COLLAPSE_MASKS=False to
restore individual mask-token selection. Grouping uses strict pairwise IoU
above the configured threshold and the same predicted class. Groups are built
in descending individual-probability order; all members must overlap every
member already admitted, avoiding transitive chains. Empty masks are not
similar. The highest-probability member represents the selected group.

Greedy inference and greedy RL references rank groups by summed probability.
This preserves total probability and EOF mass at a fixed prefix. The >50% EOF
gate therefore retains its original meaning. Query and IoU exclusions remain
unchanged after selection; masks are not averaged. If the grouping threshold
is lower than the exclusion threshold, other group members may remain eligible.
Reported mask-vs-EOF odds use the selected group mass.

Sampled RL draws categorically from groups plus EOF. Its action log probability
is the group's log-summed mass, and trajectory rescoring reconstructs the same
groups at every prefix. All member logits receive gradient credit; EOF remains
a stochastic action with its policy-gradient signal. This changes the sampled
policy when collapsing is enabled, as well as greedy selection. The group
partition and representative are discrete, with gradients through probability
sums inside the current partition. Query classes are passed from the model's
predicted semantic logits, never from GT labels.

Supervised teacher forcing continues to use original query targets. Fixed-set
RL (`OBJECT_RL_TRAIN_EOF=False`) requires collapsing disabled. The standard
free-EOF RLOO objective and ordered best-of-N fine-tuning pass class labels
through both rollout generation and likelihood scoring.

## Verification

Focused checks: `tests/test_map_state_decoder.py`, `tests/test_eof_defaults.py`,
and `tests/test_mask_groups.py`. Invoke each with `runpy.run_path` from the
repository root inside the training container. Checks cover tree/linear parity,
causal isolation, group probability and EOF conservation, complete-link grouping,
shared gradients, sampled online/rescored likelihood agreement, actual criterion
class routing for RLOO and best-of-N, and CUDA AMP generation/backpropagation.
