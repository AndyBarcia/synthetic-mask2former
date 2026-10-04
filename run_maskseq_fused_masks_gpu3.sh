#!/usr/bin/env bash
set -euo pipefail
cd /home/andy.barcia/synthetic-mask2former
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
exec apptainer exec --nv --env CUDA_VISIBLE_DEVICES=3,PYTHONUNBUFFERED=1 \
  /data/andy.barcia/fcclip-torch27-cu126.sif \
  python train_net.py --num-gpus 1 \
  --config-file output/synth_pan_R50_bs16_10kpre_2krl_per_gt_bias_fused_masks/reference_config.yaml \
  OUTPUT_DIR /home/andy.barcia/synthetic-mask2former/output/synth_pan_R50_bs16_10kpre_2krl_per_gt_bias_fused_masks \
  MODEL.MASK_FORMER.FUSED_THING_MASKS True
