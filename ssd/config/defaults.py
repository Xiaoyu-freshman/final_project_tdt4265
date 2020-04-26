'''

The comments are created by Xiaoyu Zhu at 26 April.
*This defaults.py code has been modified by Xiaoyu Zhu for TDT4265 final project.

*Additional Support:
1. Added the support for rectangle shape such as [240,320] ([Height,Weight]).
    The shape of MODEL.PRIORS.FEATURE_MAPS;
                 MODEL.PRIORS.STRIDES;
                 MODEL.PRIORS.MIN_SIZES;
                 MODEL.PRIORS.MAX_SIZES;
                 INPUT.IMAGE_SIZE;
has been changed.
2. Added the support for importing models pre-trained on ImageNet.
    Some items have been added: 
    *MODEL.BACKBONE.PRETRAINED = False 
    (when its value =True, the detector.py can import models pre-trained on ImageNet)
3. Added the support for continuing training on TDT4265 Dataset after pre-trained on Waymo Dataset.
    Some items have been added:
    *MODEL.BACKBONE.AFTER_TRAINED = False
    (when its value =True, the checkpoint.py can import models pre-trained on Waymo Data)
    *MODEL.BACKBONE.AFTER_TRAINED_File = ''
4. Added the support for more choices of Data Augmentaion.
    Some items have been added:
    *DATA_LOADER.DATA_AUGMENTATION = 'Naive'
    (The value of this item can be 'Naive', 'lufficc', 'xiaoyu', 'paper')
    *DATA_LOADER.AUGMENTATION_WEATHER = False
    (when its value =True, the Weather Augmentation will be added)
    *DATA_LOADER.AUGMENTATION_SPATIAL_LEVEL = False
    (when its value =True, the Spatial-Level Augmentation will be added)
   **To use the augmentation policy 'xiaoyu' and 'paper', a package called Albumentations has to be installed.
   (https://github.com/albumentations-team/albumentations)
     To install: pip install albumentations
5. Added the support for different optimizer.
    Some items have been added:
    *SOLVER.CHOICE = 'SGD'
    (The value of this item can be 'SGD' and 'Adam')
    
'''

from yacs.config import CfgNode as CN

cfg = CN()

cfg.MODEL = CN()
cfg.MODEL.META_ARCHITECTURE = 'SSDDetector'
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
cfg.MODEL.THRESHOLD = 0.5
cfg.MODEL.NUM_CLASSES = 21
# Hard negative mining
cfg.MODEL.NEG_POS_RATIO = 3
cfg.MODEL.CENTER_VARIANCE = 0.1
cfg.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
cfg.MODEL.BACKBONE = CN()
cfg.MODEL.BACKBONE.NAME = 'vgg'
cfg.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
cfg.MODEL.BACKBONE.PRETRAINED = False
cfg.MODEL.BACKBONE.AFTER_TRAINED = False
cfg.MODEL.BACKBONE.AFTER_TRAINED_File = ''
cfg.MODEL.BACKBONE.INPUT_CHANNELS = 3
cfg.MODEL.BACKBONE.DEPTH = 50
cfg.MODEL.BACKBONE.DROP_BLOCK = False
# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
cfg.MODEL.PRIORS = CN()
cfg.MODEL.PRIORS.FEATURE_MAPS = [[38,38], [19,19], [10,10], [5,5], [3,3], [1,1]]
cfg.MODEL.PRIORS.STRIDES = [[8,8], [16,16], [32,32], [64,64], [100,100], [300,300]]
cfg.MODEL.PRIORS.MIN_SIZES = [[30,30], [60,60], [111,111], [162,162], [213,213], [264,264]]
cfg.MODEL.PRIORS.MAX_SIZES = [[60,60], [111,111], [162,162], [213,213], [264,264], [315,315]]
cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# #boxes = 2 + #ratio * 2
cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4]  # number of boxes per feature map location
cfg.MODEL.PRIORS.CLIP = True

# -----------------------------------------------------------------------------
# Box Head
# -----------------------------------------------------------------------------
cfg.MODEL.BOX_HEAD = CN()
cfg.MODEL.BOX_HEAD.NAME = 'SSDBoxHead'
cfg.MODEL.BOX_HEAD.PREDICTOR = 'SSDBoxPredictor'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
# Image size
cfg.INPUT.IMAGE_SIZE = [300,300]
# Values to be used for image normalization, RGB layout
cfg.INPUT.PIXEL_MEAN = [123, 117, 104]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cfg.DATASETS = CN()
# List of the dataset names for training, as present in pathscfgatalog.py
cfg.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in pathscfgatalog.py
cfg.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
cfg.DATA_LOADER = CN()
# Number of data loading threads
cfg.DATA_LOADER.NUM_WORKERS = 8
cfg.DATA_LOADER.PIN_MEMORY = True
cfg.DATA_LOADER.DATA_AUGMENTATION = 'Naive'
cfg.DATA_LOADER.AUGMENTATION_WEATHER = False
cfg.DATA_LOADER.AUGMENTATION_SPATIAL_LEVEL = False

# ---------------------------------------------------------------------------- #
# Solver - The same as optimizer
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
# train configs
cfg.SOLVER.CHOICE = 'SGD'
cfg.SOLVER.MAX_ITER = 120000
cfg.SOLVER.LR_STEPS = [80000, 100000]
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.BATCH_SIZE = 32
cfg.SOLVER.LR = 1e-3
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 5e-4
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
cfg.SOLVER.WARMUP_ITERS = 500

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.TEST = CN()
cfg.TEST.NMS_THRESHOLD = 0.45
cfg.TEST.CONFIDENCE_THRESHOLD = 0.01
cfg.TEST.MAX_PER_CLASS = -1
cfg.TEST.MAX_PER_IMAGE = 100
cfg.TEST.BATCH_SIZE = 10

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.EVAL_STEP = 500 # Evaluate dataset every eval_step, disabled when eval_step < 0
cfg.MODEL_SAVE_STEP = 500 # Save checkpoint every save_step
cfg.LOG_STEP = 10 # Print logs every log_stepPrint logs every log_step
cfg.OUTPUT_DIR = "outputs"
cfg.DATASET_DIR = "datasets"