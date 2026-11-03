# YOLOX full-resolution mask supervision

Run `python train_net.py --config-file configs/synthetic/yolox_fullres.yaml`
with the synthetic renderer and panoptic evaluator installed, and the evaluator
on PYTHONPATH. This reproduces the full-resolution experiment settings:
337 proposals, 25 selection steps, batch 8, 2000 supervised iterations followed
by 200 decoder-only RL iterations. Initialization uses a fresh random seed.
Choose a fresh OUTPUT_DIR to avoid reusing existing experiment artifacts.

GT masks retain their original padded resolution for thing/stuff BCE and Dice,
assignment mask costs, and teacher-target IoU matching. Predicted stride-4
logits are bilinearly upsampled with align_corners=False. Losses use all target
pixels; no nearest-neighbor target resizing occurs. The proposal architecture
and inference resolution are unchanged. Oracle evaluation uses the complete
proposal pool before autoregressive selection and is enabled by default.

The implementation was originally developed in ignored output snapshots;
it is now available in the main source tree. Historical checkpoints and their
source snapshots are preserved. Native model/oracle contracts and a single-pixel
supervision/backpropagation regression test cover the change.
