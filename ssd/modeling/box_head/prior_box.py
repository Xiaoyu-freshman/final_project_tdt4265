'''

The comments are created by Xiaoyu Zhu at 26 April.
*This prior_box code has been modified by Xiaoyu Zhu for TDT4265 final project.
*with the referencing of the discussions on Piazza.

*Additional Support:
1. Added the support for rectangle shape such as [240,320] ([Height,Weight]).

'''


import torch
from math import sqrt
from itertools import product


class PriorBox:
    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE #[240,320] [H,W]
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        #print('self.feature_maps',self.feature_maps)
        for k, f in enumerate(self.feature_maps):
            #Stride:  [[8,8], [16,16], [30,32], [60,64], [120,106], [240,320]]
            scale_x = self.image_size[1] / self.strides[k][1] #scale_x means the witdh  320/stride_W
            scale_y = self.image_size[0] / self.strides[k][0] #scale_y means the height 240/stride_H
            for i, j in product(range(f[0]), range(f[1])):
                # unit center x,y
                #cuz the shape change from [300,300] to [240,300]
                #so the central point should be modified
                cx = (j + 0.5) / scale_x #j means the width and corespond to cx
                cy = (i + 0.5) / scale_y #i means the height and corespond to cy

                # small sized square box
                size_h = self.min_sizes[k][0]
                size_w = self.min_sizes[k][1]
                h = size_h/ self.image_size[0]
                w = size_w/ self.image_size[1]
                priors.append([cx, cy, w, h])

                # big sized square box
                size_h = sqrt(self.min_sizes[k][0] * self.max_sizes[k][0])
                size_w = sqrt(self.min_sizes[k][1] * self.max_sizes[k][1])
                h = size_h/ self.image_size[0]
                w = size_w/ self.image_size[1]
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size_h = self.min_sizes[k][0]
                size_w = self.min_sizes[k][1]
                h = size_h/ self.image_size[0]
                w = size_w/ self.image_size[1]
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
