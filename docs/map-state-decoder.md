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
