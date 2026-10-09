# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # Online CUDA synthetic-scene renderer.
    cfg.INPUT.SYNTHETIC_SCENE = CN()
    cfg.INPUT.SYNTHETIC_SCENE.WIDTH = 768
    cfg.INPUT.SYNTHETIC_SCENE.HEIGHT = 512
    cfg.INPUT.SYNTHETIC_SCENE.SEED = 1234
    cfg.INPUT.SYNTHETIC_SCENE.HOUSE_COUNT = 10
    cfg.INPUT.SYNTHETIC_SCENE.TREE_COUNT = 10
    cfg.INPUT.SYNTHETIC_SCENE.CLOUD_COUNT = 5
    cfg.INPUT.SYNTHETIC_SCENE.CAR_COUNT = 5
    cfg.INPUT.SYNTHETIC_SCENE.PERSON_COUNT = 5
    cfg.INPUT.SYNTHETIC_SCENE.TEST_SEED = 4321
    cfg.INPUT.SYNTHETIC_SCENE.TEST_SAMPLES = 1000
    cfg.INPUT.SYNTHETIC_SCENE.TEST_BATCH_SIZE = 32

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    # Image-conditioned prior for selecting object queries. The same Q logits
    # are used by every decoder layer for matching and binary supervision.
    cfg.MODEL.MASK_FORMER.QUERY_BIAS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.QUERY_BIAS_MATCHER_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_LOSS_TYPE = "point"
    # Extra mask-only union target for each thing class with multiple instances.
    cfg.MODEL.MASK_FORMER.FUSED_THING_MASKS = True

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    # Number of suffix permutations sharing a randomly sampled training prefix.
    cfg.MODEL.MASK_FORMER.OBJECT_DEC_PREFIX_TREE_BRANCHES = 4
    cfg.MODEL.MASK_FORMER.OBJECT_PRETRAIN_PQ_TEMPERATURE = 0.0
    cfg.MODEL.MASK_FORMER.OBJECT_PRETRAIN_PQ_REWARD = "terminal"
    cfg.MODEL.MASK_FORMER.OBJECT_DEC_LAYERS = 5
    # NoPE by default; learned absolute positions and RoPE remain opt-in.
    cfg.MODEL.MASK_FORMER.OBJECT_DEC_POSITION_ENCODING = False
    cfg.MODEL.MASK_FORMER.OBJECT_DEC_ROPE = False
    cfg.MODEL.MASK_FORMER.OBJECT_RL_WEIGHT = 0.0
    cfg.MODEL.MASK_FORMER.OBJECT_RL_ONLY = False
    cfg.MODEL.MASK_FORMER.OBJECT_RL_MAX_STEPS = 101
    cfg.MODEL.MASK_FORMER.OBJECT_RL_NUM_SAMPLES = 4
    cfg.MODEL.MASK_FORMER.OBJECT_RL_BASELINE = "rloo_greedy"
    # policy_gradient uses RLOO/SCST; Best-of-N variants imitate a trajectory
    # (best_of_n_ft) or its unordered selected-query set (best_of_n_set).
    cfg.MODEL.MASK_FORMER.OBJECT_RL_OBJECTIVE = "policy_gradient"
    cfg.MODEL.MASK_FORMER.OBJECT_RL_TRAIN_EOF = True
    # Zero scores at ground-truth resolution; positive values use a square proxy.
    cfg.MODEL.MASK_FORMER.OBJECT_RL_REWARD_SIZE = 0

    # Exclude selected masks and other vocabulary masks above this predicted-mask IoU.
    cfg.MODEL.MASK_FORMER.OBJECT_DEC_MASK_IOU_THRESHOLD = 0.8
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_PAINT_ORDER = "forward"
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # Sequential training phases; RL solver settings apply only to the RL phase.
    cfg.TRAINING = CN()
    cfg.TRAINING.MODE = "supervised"
    cfg.TRAINING.RL = CN()
    cfg.TRAINING.RL.MAX_ITER = 10000
    cfg.TRAINING.RL.BASE_LR = 0.00001
    cfg.TRAINING.RL.WARMUP_ITERS = 0
    cfg.TRAINING.RL.STEPS = (8000, 9500)
    cfg.TRAINING.RL.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.TRAINING.RL.WEIGHT = 1.0
