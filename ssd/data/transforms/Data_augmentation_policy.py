#------------A little function to conver corrodinate-----------
'''
Defined by Xiaoyu
For use this augmentation method, since the order of the element in bboxes is
min_y, min_x, max_y, max_x

'''
from urllib.request import urlopen
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

import albumentations as A

def get_aug(aug, min_area=0., min_visibility=0.):
    
    return A.Compose(aug, bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area,min_visibility=min_visibility, label_fields=['category_id']))


class  DataAaugmentationPolicy(object):
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
        #Color_Level Change
        if cfg.DATA_LOADER.AUGMENTATION_WEATHER:
            trans_color_level = A.Compose([
                A.Cutout(num_holes=20, max_h_size=64, max_w_size=64, fill_value=255, always_apply=False, p=1),
                A.Equalize(p=1),
                A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, always_apply=False, p=1),
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.08, always_apply=False, p=1),
                    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=1),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=1),
                    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=1)
                ]),
            ])
        else:
            trans_color_level = A.Compose([
                A.Cutout(num_holes=20, max_h_size=64, max_w_size=64, fill_value=255, always_apply=False, p=1),
                A.Equalize(p=1),
                A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, always_apply=False, p=1),
            ])
        #Spatial_Level
        if cfg.DATA_LOADER.AUGMENTATION_SPATIAL_LEVEL:
            trans_rotate_level = A.Compose([
                A.OneOf([
                    A.Rotate(limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1),
                    A.RandomRotate90(always_apply=False, p=1),
                    A.VerticalFlip(always_apply=False, p=1), 
                    A.HorizontalFlip(always_apply=False, p=1)
                ]),

            ])
        #Apply the trans
        aug=get_aug(trans_color_level)
        augmented = aug(**annotations)
        img=augmented['image']
        bbox=augmented['bboxes']
        bbox = np.array(bbox)
        label=augmented['category_id']
        #try rotate
        if cfg.DATA_LOADER.AUGMENTATION_SPATIAL_LEVEL:
            aug1 = get_aug(trans_rotate_level)
            augmented1 = aug1(**augmented)
            img1=augmented1['image']
            bbox1=augmented1['bboxes']
            bbox1 = np.array(bbox1)
            label1=augmented1['category_id']
            #if rotate fail
            if bbox1.shape[0] == 0: 
                return img, bbox.astype(np.float32) , np.array(label)
            else:
                return img1, bbox1.astype(np.float32) , np.array(label1)
        else:
            return img, bbox.astype(np.float32) , np.array(label)
    
    
    
    
    
   