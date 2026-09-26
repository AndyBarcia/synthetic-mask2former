#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="0,3"
output_dir="$PWD/output/synthetic_panoptic_object_decoder_rl_vocab_attention"
checkpoint="$PWD/output/synthetic_panoptic_bs16_10k_global_hun_bias_mask5_object5_vocab_attention/model_final.pth"
mkdir -p "$output_dir"

exec apptainer exec --nv --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  /data/andy.barcia/fcclip-torch27-cu126.sif \
  python train_net.py --num-gpus 2 \
  --config-file configs/synthetic-scene/panoptic-segmentation/maskformer2_R50_bs16.yaml \
  MODEL.WEIGHTS "$checkpoint" \
  MODEL.MASK_FORMER.DEC_LAYERS 6 \
  MODEL.MASK_FORMER.OBJECT_DEC_LAYERS 5 \
  MODEL.MASK_FORMER.OBJECT_RL_ONLY True \
  MODEL.MASK_FORMER.OBJECT_RL_WEIGHT 1.0 \
  MODEL.MASK_FORMER.OBJECT_RL_MAX_STEPS 100 \
  MODEL.MASK_FORMER.OBJECT_RL_REWARD_SIZE 0 \
  MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD 0.8 \
  MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD 0.8 \
  SOLVER.IMS_PER_BATCH 16 \
  SOLVER.MAX_ITER 10000 \
  SOLVER.BASE_LR 0.00001 \
  SOLVER.LR_SCHEDULER_NAME WarmupMultiStepLR \
  SOLVER.STEPS '(8000,9500)' \
  SOLVER.CHECKPOINT_PERIOD 1000 \
  INPUT.SYNTHETIC_SCENE.TEST_BATCH_SIZE 32 \
  OUTPUT_DIR "$output_dir" \
  "$@"
