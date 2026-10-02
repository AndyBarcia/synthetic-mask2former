# Supervised and object-decoder RL training

`train_net.py` supports three modes. Existing commands use `supervised` by default.

```bash
# Normal supervised training (existing SOLVER settings).
python train_net.py --num-gpus 1 \
  --config-file configs/synthetic-scene/panoptic-segmentation/maskformer2_R50_bs16.yaml \
  TRAINING.MODE supervised OUTPUT_DIR output/supervised

# Supervised training followed automatically by decoder RL.
python train_net.py --num-gpus 1 \
  --config-file configs/synthetic-scene/panoptic-segmentation/maskformer2_R50_bs16.yaml \
  TRAINING.MODE supervised_then_rl \
  SOLVER.MAX_ITER 10000 \
  TRAINING.RL.MAX_ITER 10000 TRAINING.RL.BASE_LR 0.00001 \
  OUTPUT_DIR output/supervised_then_rl

# Decoder RL starting from an existing compatible checkpoint.
python train_net.py --num-gpus 1 \
  --config-file configs/synthetic-scene/panoptic-segmentation/maskformer2_R50_bs16.yaml \
  TRAINING.MODE rl MODEL.WEIGHTS output/supervised/model_final.pth \
  TRAINING.RL.MAX_ITER 10000 TRAINING.RL.BASE_LR 0.00001 \
  OUTPUT_DIR output/rl
```

Use your usual environment/container launcher. Model dimensions, decoder layers,
query count, and class count must match the source checkpoint. The checkpoint
should include a trained object decoder.

The sequential mode writes separate `supervised/` and `rl/` directories under
`OUTPUT_DIR`. It loads `supervised/model_final.pth` as model weights, then creates
a fresh RL optimizer, scheduler, and iteration counter. RL-only mode uses
`OUTPUT_DIR` directly. Choose a new output directory for a new experiment.

Add `--resume` to the same command to restore optimizer, scheduler, and iteration
from the phase's `last_checkpoint`. Sequential mode resumes RL if its checkpoint
exists; otherwise it resumes supervised training or starts RL from the completed
supervised checkpoint. Without `--resume`, the command starts a new run. Keep
`MODEL.WEIGHTS` set when resuming RL-only mode.

RL freezes the backbone and segmentation head, runs them in evaluation mode,
and updates only the object decoder. Supervised training retains prefix-tree
teacher forcing and the current matcher and inference behavior.

RL phase solver overrides are `TRAINING.RL.MAX_ITER`, `BASE_LR`, `WARMUP_ITERS`,
`STEPS`, and `LR_SCHEDULER_NAME`. Other solver settings, including batch size,
checkpoint period, optimizer type, and AMP, are shared. Defaults are 10,000 RL
iterations, learning rate 1e-5, no warmup, and a multistep schedule at 8,000 and
9,500. Adjust `TRAINING.RL.STEPS` when changing the iteration budget.

`TRAINING.RL.WEIGHT` controls the RL loss scale (default 1). The mode sets
`MODEL.MASK_FORMER.OBJECT_RL_ONLY` and `OBJECT_RL_WEIGHT` automatically.

The default RL objective is policy gradient with an RLOO plus greedy baseline,
four sampled trajectories per image, and free decoding including EOF. Configure
it through `MODEL.MASK_FORMER.OBJECT_RL_*`:

- `NUM_SAMPLES`: number of sampled trajectories.
- `BASELINE`: `rloo_greedy` or `scst`.
- `TRAIN_EOF`: default `True`; `False` learns ordering among query-bias proposals
  using the ground-truth object count during training.
- `MAX_STEPS`: default 101; EOF training requires at least query count plus one.
- `REWARD_SIZE`: zero uses ground-truth resolution; positive values use a square
  proxy resolution.
- `OBJECTIVE`: `policy_gradient`, `best_of_n_ft`, or `best_of_n_set`. The latter
  two imitate the best sampled trajectory or set and require EOF training.

Rewards use per-image mean class PQ and the configured panoptic paint order.
RL diagnostics appear in the standard metrics files and console logs.

For evaluation, point `MODEL.WEIGHTS` to the desired phase checkpoint and use
`--eval-only`. In sequential mode, use the phase's output directory if you also
use `--resume` for evaluation.

The query-bias head encodes each GT mask's pooled pixel-decoder features and
relative area into a `(GT, Q)` matrix. Hungarian matching uses the transpose of
its sigmoid probabilities as a negative cost. The bias loss marks matched
GT/query pairs positive and weights all other pairs by `NO_OBJECT_WEIGHT`.
The same matrix is reused at every auxiliary decoder depth. At inference, each
predicted soft mask is encoded and its corresponding query logit supplies the
existing per-query confidence score. The new encoder replaces the old global
linear bias head; its parameters must be trained when loading older checkpoints.
