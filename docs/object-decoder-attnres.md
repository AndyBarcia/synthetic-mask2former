# Object decoder attention residuals

Enable with `MODEL.MASK_FORMER.OBJECT_DEC_VARIANT attnres` during training
and evaluation. The default `baseline` preserves the original decoder.

Full attention residuals aggregate the initial token states and raw previous
sublayer outputs. Each aggregation has a zero-initialized learned query and
RMS-normalized depth keys; attention weights normalize across depth. There are
four aggregations per decoder layer (self-attention, image cross-attention,
feed-forward, mask cross-attention) and one final aggregation. Transformations
use PreNorm. Mask embeddings, pointer output scoring, and generation are unchanged.
Existing AttnRes object decoder checkpoints retain compatible parameter names.
Baseline checkpoints require training the additional AttnRes parameters.

Run `python tests/test_object_decoder_attnres.py` for initialization, depth
aggregation, causal-prefix, packed-tree equivalence, and gradient checks.
