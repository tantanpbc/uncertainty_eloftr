from yacs.config import CfgNode as CN
_CN = CN()

##############  ↓  LoFTR Pipeline  ↓  ##############
_CN.LOFTR = CN()
_CN.LOFTR.BACKBONE_TYPE = 'RepVGG'
_CN.LOFTR.ALIGN_CORNER = False
_CN.LOFTR.RESOLUTION = (8, 1)
_CN.LOFTR.FINE_WINDOW_SIZE = 8  # window_size in fine_level, must be even
_CN.LOFTR.MP = False
_CN.LOFTR.REPLACE_NAN = False
_CN.LOFTR.EVAL_TIMES = 1
_CN.LOFTR.HALF = False

# 1. LoFTR-backbone (local feature CNN) config
_CN.LOFTR.BACKBONE = CN()
_CN.LOFTR.BACKBONE.BLOCK_DIMS = [64, 128, 256]  # s1, s2, s3

# 2. LoFTR-coarse module config
_CN.LOFTR.COARSE = CN()
_CN.LOFTR.COARSE.D_MODEL = 256
_CN.LOFTR.COARSE.D_FFN = 256
_CN.LOFTR.COARSE.NHEAD = 8
_CN.LOFTR.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.LOFTR.COARSE.AGG_SIZE0 = 4
_CN.LOFTR.COARSE.AGG_SIZE1 = 4
_CN.LOFTR.COARSE.NO_FLASH = False
_CN.LOFTR.COARSE.ROPE = True
_CN.LOFTR.COARSE.NPE = None  # [832, 832, long_side, long_side]

# 3. Coarse-Matching config
_CN.LOFTR.MATCH_COARSE = CN()
_CN.LOFTR.MATCH_COARSE.THR = 0.2
_CN.LOFTR.MATCH_COARSE.BORDER_RM = 2
_CN.LOFTR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2
_CN.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200
_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = True
_CN.LOFTR.MATCH_COARSE.SKIP_SOFTMAX = False
_CN.LOFTR.MATCH_COARSE.FP16MATMUL = False

# 4. Fine-Matching config
_CN.LOFTR.MATCH_FINE = CN()
_CN.LOFTR.MATCH_FINE.SPARSE_SPVS = True
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 1.0
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 8
# ── NEW: uncertainty prediction ──────────────────────────────────────────────
# False → log-variance derived for free from heatmap spatial entropy (recommended)
# True  → small learned linear head on the feature slice (add when ECE stays high)
_CN.LOFTR.MATCH_FINE.PREDICT_VAR = False

# 5. LoFTR Losses
# -- # coarse-level
_CN.LOFTR.LOSS = CN()
_CN.LOFTR.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.LOFTR.LOSS.COARSE_WEIGHT = 1.0
_CN.LOFTR.LOSS.COARSE_SIGMOID_WEIGHT = 1.0
_CN.LOFTR.LOSS.LOCAL_WEIGHT = 0.5
_CN.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT = False
_CN.LOFTR.LOSS.FINE_OVERLAP_WEIGHT = False
_CN.LOFTR.LOSS.FINE_OVERLAP_WEIGHT2 = False
# -- - -- # focal loss (coarse)
_CN.LOFTR.LOSS.FOCAL_ALPHA = 0.25
_CN.LOFTR.LOSS.FOCAL_GAMMA = 2.0
_CN.LOFTR.LOSS.POS_WEIGHT = 1.0
_CN.LOFTR.LOSS.NEG_WEIGHT = 1.0
# -- # fine-level
_CN.LOFTR.LOSS.FINE_TYPE = 'l2_with_std'  # ['l2_with_std', 'l2']
_CN.LOFTR.LOSS.FINE_WEIGHT = 1.0
_CN.LOFTR.LOSS.FINE_CORRECT_THR = 1.0
# ── NEW: sub-pixel local regression loss type ────────────────────────────────
# 'l2'           → original L2 loss, no uncertainty (default, backward-compatible)
# 'nll_gaussian' → Heteroscedastic Gaussian NLL using predicted log-variance
# 'nll_laplace'  → Heteroscedastic Laplace NLL using predicted log-variance
_CN.LOFTR.LOSS.LOCAL_LOSS_TYPE = 'l2'


##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None
_CN.DATASET.VAL_INTRINSIC_PATH = None
_CN.DATASET.FP16 = False
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

# ScanNet options
_CN.DATASET.SCAN_IMG_RESIZEX = 640
_CN.DATASET.SCAN_IMG_RESIZEY = 480

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 640
_CN.DATASET.MGDPT_IMG_PAD = True
_CN.DATASET.MGDPT_DEPTH_PAD = True
_CN.DATASET.MGDPT_DF = 8

_CN.DATASET.NPE_NAME = None

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None
_CN.TRAINER.FIND_LR = False

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None
_CN.TRAINER.ADAM_DECAY = 0.
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30
_CN.TRAINER.ELR_GAMMA = 0.999992

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, LO-RANSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # ['scene_balance', 'random', 'normal']
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True
_CN.TRAINER.SB_SUBSET_SHUFFLE = True
_CN.TRAINER.SB_REPEAT = 1
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _CN.clone()