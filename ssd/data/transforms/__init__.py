'''

The comments are created by Xiaoyu Zhu at 26 April.
*This __init__.py code has been modified by Xiaoyu Zhu for TDT4265 final project.
*with the referencing of :
1. Lufficc's SSD code: https://github.com/lufficc/SSD/blob/master/ssd/data/transforms/__init__.py
2. Albumentations official guide: https://github.com/albumentations-team/albumentations#pypi
3. Paper: Learning Data Augmentation Strategies for Object Detection

*There are 4 choices of augmentation polices: 
    1. 'Naive': used when pre-training the final model on Waymo Dataset.
    2. 'lufficc': borrowed from https://github.com/lufficc/SSD/blob/master/ssd/data/transforms/__init__.py
    3. 'xiaoyu': A private approach for data augmentation based on the intuition from the paper.
    4. 'paper': A private way for implementing the polices mentioned in the paper.

'''

from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *
from .Data_augmentation_policy import DataAaugmentationPolicy
from .Data_augmentation_paper import Augment_paper

def build_transforms(cfg, is_train=True):
    if is_train:
        policy = cfg.DATA_LOADER.DATA_AUGMENTATION
        if policy == 'Naive':
            transform = [
                ConvertFromInts(),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),]
        elif policy == 'lufficc':
            transform = [
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(cfg.INPUT.PIXEL_MEAN),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(cfg.INPUT.IMAGE_SIZE),
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),]
        elif policy == 'paper':
            transform = [
                Augment_paper(cfg),
                ConvertFromInts(),
                ToPercentCoords(), 
                Resize(cfg.INPUT.IMAGE_SIZE), #Resize need topercent fistly.
                SubtractMeans(cfg.INPUT.PIXEL_MEAN),
                ToTensor(),]
            
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
