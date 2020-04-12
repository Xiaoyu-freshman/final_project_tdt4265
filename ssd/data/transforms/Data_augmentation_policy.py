#------------A little function to conver corrodinate-----------
'''
Defined by Xiaoyu
For use this augmentation method, since the order of the element in bboxes is
min_y, min_x, max_y, max_x

'''
from .autoaugment_utils import *


def trans_coor_boxes(box_original):
#covnert x_min,y_min,x_max,y_max to min_y, min_x, max_y, max_x
    box_trans=np.zeros(box_original.shape)
    box_trans[:,0]=box_original[:,1]
    box_trans[:,1]=box_original[:,0]
    box_trans[:,2]=box_original[:,3]
    box_trans[:,3]=box_original[:,2]
    return box_trans

class  DataAaugmentationPolicy(object):
    def __call__(self, image, boxes, labels=None):
        image, boxes = distort_image_with_autoaugment(img, trans_coor_boxes(box), 'v1')
        boxes=trans_coor_boxes(box)
        
        return image, boxes, labels
         
    
    
    
    
    
   