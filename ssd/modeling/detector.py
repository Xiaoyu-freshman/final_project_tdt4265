import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from ssd.modeling.backbone.vgg import VGG
from ssd.modeling.backbone.basic import BasicModel
from ssd.modeling.box_head.box_head import SSDBoxHead
from ssd.utils.model_zoo import load_state_dict_from_url
from ssd.modeling.backbone.resnet import ResNet
from torchvision.models.resnet import BasicBlock,Bottleneck
#from ssd.modeling.backbone.resnet_base import resnet_base#ResNetBase,BasicBlock,Bottleneck
from ssd import torch_utils

class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = SSDBoxHead(cfg)
        print(
            "Detector initialized. Total Number of params: ",
            f"{torch_utils.format_params(self)}")
        print(
            f"Backbone number of parameters: {torch_utils.format_params(self.backbone)}")
        print(
            f"SSD Head number of parameters: {torch_utils.format_params(self.box_head)}")

    def forward(self, images, targets=None):
        features = self.backbone(images)

        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections


def build_backbone(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    print(backbone_name)
    if backbone_name == "basic":
        model = BasicModel(cfg)
        return model
    if backbone_name == "vgg":
        model = VGG(cfg)
        if cfg.MODEL.BACKBONE.PRETRAINED:
            state_dict = load_state_dict_from_url(
                "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth")
            model.init_from_pretrain(state_dict)
    if backbone_name == "resnet":
        #model = ResNet(cfg)
        depth = cfg.MODEL.BACKBONE.DEPTH
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',}
        name_dict = {18: 'resnet18', 34: 'resnet34', 50: 'resnet50', 101: 'resnet101', 152: 'resnet152'}
        layers_dict = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 
                       101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        block_dict = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck}
        model = ResNet(cfg, block_dict[depth], layers_dict[depth])
        if cfg.MODEL.BACKBONE.PRETRAINED:
            pretrained_dict = model_zoo.load_url(model_urls[name_dict[depth]])
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        return model
