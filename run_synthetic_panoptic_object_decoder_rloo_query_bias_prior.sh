#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=3

output_dir=/home/andy.barcia/synthetic-mask2former/output/synthetic_panoptic_object_decoder_rloo_query_bias_order_prior_eof_forward_paint
checkpoint=/home/andy.barcia/synthetic-mask2former/output/synthetic_panoptic_bs16_10k_global_hun_bias_mask5_object5_vocab_attention_query_bias_order_prior/model_final.pth
dataset_source=/home/andy.barcia/synthetic-mask2former/mask2former/data/datasets/synthetic-scene
mkdir -p "$output_dir"

exec apptainer exec --nv \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env PYTHONPATH="$dataset_source" \
  /data/andy.barcia/fcclip-torch27-cu126.sif \
  python train_net.py --num-gpus 1 --resume \
  --config-file configs/synthetic-scene/panoptic-segmentation/maskformer2_R50_bs16.yaml \
  MODEL.WEIGHTS "$checkpoint" \
  MODEL.MASK_FORMER.DEC_LAYERS 6 \
  MODEL.MASK_FORMER.OBJECT_DEC_LAYERS 5 \
  MODEL.MASK_FORMER.OBJECT_DEC_BIAS_ORDER_CONSTRAINT True \
  MODEL.MASK_FORMER.OBJECT_RL_ONLY True \
  MODEL.MASK_FORMER.OBJECT_RL_WEIGHT 1.0 \
  MODEL.MASK_FORMER.OBJECT_RL_NUM_SAMPLES 64 \
  MODEL.MASK_FORMER.OBJECT_RL_OBJECTIVE policy_gradient \
  MODEL.MASK_FORMER.OBJECT_RL_BASELINE rloo_greedy \
  MODEL.MASK_FORMER.OBJECT_RL_TRAIN_EOF True \
  MODEL.MASK_FORMER.OBJECT_RL_MAX_STEPS 101 \
  MODEL.MASK_FORMER.OBJECT_RL_REWARD_SIZE 0 \
  MODEL.MASK_FORMER.TEST.OBJECT_DECODER_MODE eof \
  MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD 0.8 \
  MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD 0.8 \
  MODEL.MASK_FORMER.TEST.PANOPTIC_PAINT_ORDER forward \
  SOLVER.IMS_PER_BATCH 16 \
  SOLVER.MAX_ITER 10000 \
  SOLVER.BASE_LR 0.00001 \
  SOLVER.LR_SCHEDULER_NAME WarmupMultiStepLR \
  SOLVER.STEPS '(8000,9500)' \
  SOLVER.CHECKPOINT_PERIOD 1000 \
  INPUT.SYNTHETIC_SCENE.TEST_BATCH_SIZE 32 \
  OUTPUT_DIR "$output_dir" \
  "$@"
