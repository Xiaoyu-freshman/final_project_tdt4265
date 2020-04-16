from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *
from .Data_augmentation_policy import DataAaugmentationPolicy


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            #ConvertFromInts(),
#             DataAaugmentationPolicy(),
#             ToPercentCoords(), 
#             Resize(cfg.INPUT.IMAGE_SIZE), #Resize need topercent fistly.
#             SubtractMeans(cfg.INPUT.PIXEL_MEAN),
#             ToTensor(),
            ConvertFromInts(),
            #Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
