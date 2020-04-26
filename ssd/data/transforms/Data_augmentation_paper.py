'''

The comments are created by Xiaoyu Zhu at 26 April.
*This Data_augmentation_paper.py code has been created by Xiaoyu Zhu for TDT4265 final project.
*with the referencing of :
1. Albumentations official guide: https://github.com/albumentations-team/albumentations#pypi
2. Paper: Learning Data Augmentation Strategies for Object Detection
*Functions:
1. get_aug() borrowed from Albumentations official guide

*Additional Support:
1. Added the private way of implementing the polices mentioned in the paper. Because of Albumentations doesn't have the operation which only changes bounding boxes, there are only color-level and spatial-level operation. 
I tried to use RandomSizedBBoxSafeCrop() to take place the only-bbox-level operation.
2. Added the support of Weather Operation, which is really cool.
    (it works when self.cfg.DATA_LOADER.AUGMENTATION_WEATHER= True)

'''
from urllib.request import urlopen
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

import albumentations as A

def get_aug(aug, min_area=0., min_visibility=0.):
    
    return A.Compose(aug, bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area,min_visibility=min_visibility, label_fields=['category_id']))


class  Augment_paper(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, image, boxes=None, labels=None):
        #initialize the format for lib albumentations
        if boxes.shape[0] == 0:
            return image, boxes, labels
        bbox=[]
        for i in boxes:
            bbox.append(list(i))
        #create annotations
        annotations = {'image': image, 'bboxes': boxes, 'category_id':  list(labels)}
        #create translation
        sub1=A.Compose([
            A.HorizontalFlip(always_apply=False, p=0.6),
            A.Equalize(p=0.8),
        ])
        sub2=A.Compose([
            A.VerticalFlip(always_apply=False, p=0.5),
            A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=255, always_apply=False, p=0.8),
        ])
        sub3=A.Compose([
            A.OneOf([
                A.RandomSizedBBoxSafeCrop(720, 960, erosion_rate=0.0, interpolation=1, always_apply=False, p=0.5),
                A.RandomSizedBBoxSafeCrop(480, 640, erosion_rate=0.0, interpolation=1, always_apply=False, p=0.5),
                A.RandomSizedBBoxSafeCrop(240, 320, erosion_rate=0.0, interpolation=1, always_apply=False, p=0.5),
            ]),
            A.VerticalFlip(always_apply=False, p=0.6),

        ])
        sub4=A.Compose([
            A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.6),
            A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, always_apply=False, p=1),
            
        ])
        
        
        if self.cfg.DATA_LOADER.AUGMENTATION_WEATHER:
            trans_color_level = A.Compose([
                A.OneOf([
                    sub1,
                    sub2,
                    sub3,
                    sub4
                ]),
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.08, always_apply=False, p=0.5),
                    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=0.5),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=0.5),
                    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.5)
                ])
            ])
        else:
            trans_color_level = A.OneOf([
                sub1,
                sub2,
                sub3,
                sub4
            ])

        #Apply the trans
        aug=get_aug(trans_color_level)
        augmented = aug(**annotations)
        img=augmented['image']
        bbox=augmented['bboxes']
        bbox = np.array(bbox)
        label=augmented['category_id']
        #When bbox dispearing, just give up this time.
        if bbox.shape[0] == 0: 
            return image, boxes, labels
        else:
            return img, bbox.astype(np.float32) , np.array(label)
    
    
    
    
    
   