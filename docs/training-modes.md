# Supervised and object-decoder RL training

`train_net.py` supports five modes. Existing commands use `supervised` by default.

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
checkpoint period, optimizer type, and AMP, are shared. Defaults are 20,000 RL
iterations, learning rate 1e-5, no warmup, and a multistep schedule at 16,000 and
19,000. Adjust `TRAINING.RL.STEPS` when changing the iteration budget.

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

Rewards use distributed batch PQ by default and the configured panoptic paint
order. Use OBJECT_RL_REWARD image_pq for per-image mean class PQ.
RL diagnostics appear in the standard metrics files and console logs.
When duplicate masks are collapsed, sampled trajectories retain the exact
complete-link group memberships and representative indices. Gradient rescoring
reuses that detached topology while recomputing differentiable group probability
masses. This preserves sampled action support even when nearly tied logits round
differently during incremental sampling and batched rescoring.

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

Set `MODEL.MASK_FORMER.OBJECT_RL_REWARD batch_pq` to use distributed
batch PQ counterfactual rewards. Each sampled image replaces its greedy
prediction in the full batch's per-class IoU/TP/FP/FN totals. Its reward is
the resulting change in mean class PQ, multiplied by the global batch size
(to compensate for the policy loss's image mean). The greedy reward is zero.
This accounts for both class frequency and changes in the PQ denominator;
`batch_pq` is the default. Batch PQ is still a finite-batch proxy for
dataset PQ, and training augmentation/resolution differences still apply.
In this mode `rl/reward/*_scaled_delta_pq` logs signed PQ changes multiplied
by global batch size. Greedy has zero relative reward by definition.
`rl/pq/greedy_batch_pq` and `rl/pq/sample_batch_pq` separately report absolute
PQ (0–100) for the greedy batch and the mean of complete sampled batches.

Set `MODEL.MASK_FORMER.OBJECT_RL_REWARD batch_rq` to optimize distributed
batch recognition quality: mean per-class `TP / (TP + 0.5 * (FP + FN))`.
It uses the same greedy-image counterfactual replacement and global-batch-size
scaling as batch PQ, with the same strict IoU matching. Rewards and advantages
are logged as RQ; absolute batch PQ and RQ are both logged for comparison.

Fused mask supervision uses only query slots left after all original targets.
If unions would exceed the query count, retain unions in thing-class order up
to that budget; preserve every original target.

`TRAINING.MODE hybrid` starts from `MODEL.WEIGHTS` and combines RL with supervised
segmentation and query-bias losses, including auxiliary losses. Object-decoder
teacher forcing is excluded; the object decoder learns only from RL. The backbone,
pixel decoder, mask decoder, and query-bias head remain trainable. Policy inputs
retain their gradients; sampling, hard mask thresholds, and rewards do not.
`supervised_then_hybrid` hands off supervised weights to a fresh hybrid optimizer
and scheduler in `hybrid/`, using the `TRAINING.RL` schedule and loss weight.

In hybrid policy-gradient training, set
`MODEL.MASK_FORMER.HYBRID_SELECTED_PANOPTIC_SUPERVISION True` to replace
all-proposal supervision with supervision of the best sampled panoptic map
(including greedy as a candidate, chosen by the configured reward). Hungarian
matching uses visible hard painted segments; class, mask, and query-bias losses
use the corresponding selected predictions. Stuff queries merge by predicted
class; their soft masks use differentiable last-wins compositing and their class
and GT-conditioned query-bias logits are averaged. Occluded segments and unused
proposals receive no direct supervised target. Auxiliary layers reuse the final
assignment. This mode excludes extra fused targets and object-decoder teacher
forcing, and does not add coverage supervision for omitted GT objects.

To reproduce the reference 80k supervised + 20k selected-map RQ hybrid recipe:

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/coco/panoptic-segmentation/maskformer2_R50_bs16_80kpre_20khybrid_selected_rq.yaml \
  OUTPUT_DIR output/coco_80kpre_20khybrid_reproduction
```

`supervised_then_hybrid` disables selected-map supervision
flags for pretraining. The supervised phase keeps all-proposal Hungarian
matching, raw-mask losses, per-layer auxiliary matching, and object-decoder
teacher forcing. Only the hybrid phase enables the configured selected-map
matching and painted-mask losses. The handoff loads supervised model weights into a fresh optimizer
and scheduler. Hybrid LR drops at 16k/19k local iterations (96k/99k total).
`--resume` restores the appropriate phase checkpoint. This reproduces the
training recipe, not identical random sampling or the reference run's GPU
preemption and change from two GPUs to one.

For evaluation, the reproduction config can be used with `--eval-only` and an
explicit `MODEL.WEIGHTS`. Training-only objectives are disabled without changing
checkpoint or output paths. Class/query-bias losses average per image; selected
mask losses normalize by the distributed matched-segment count.
