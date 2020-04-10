import torch.nn as nn
import torch.nn.functional as F
import torch
from ssd.utils import box_utils
import pdb

class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.gamma = 2.0
        self.alpha = 0.25
    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        #pdb.set_trace()
        num_classes = confidence.size(2) #class = 5
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]  #0 is the bcakground class
            
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio) #mask is to conduct the hear_negative_mining for reducing the negative labels.
        confidence = confidence[mask, :] #In the frist loop, the confidence.shape is torch.Size([3108,5])
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum') #.view(-1,c)'s -1 means that the number of row can be any number since you are not sure!
###--------------try to modify cross-entropy into focal loss--------------------------
        p = torch.exp(-classification_loss)
        focal_loss = (1-p)**self.gamma*classification_loss
###----------------modefied by Xiaoyu------------------------------------------------
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, focal_loss/ num_pos #classification_loss / num_pos
