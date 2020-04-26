'''

The comments are created by Xiaoyu Zhu at 26 April.
*This backbone code is designed by Xiaoyu Zhu for TDT4265 final project.
*with the referencing of :
1. Torch official resnet code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. ssjatmhmy's Resnet+SSD code (This is unfinished code): https://github.com/ssjatmhmy/pytorch-resnet-ssd/blob/master/resnet_base.py

*Functions:
1. _make_layer(): borrowed from Torch official resnet code.
2. _make_extra_layers(): designed by Xiaoyu, which is the naive double-layer Conv2d. Not used in the final structure.
3. kaiming weight normal borrowed from ssjatmhmy's Resnet+SSD code
*Additional Support:
1. Added the support of DropBlock, with referencing of miguelvr/dropblock (https://github.com/miguelvr/dropblock/blob/master/examples/resnet-cifar10.py)
*To install it: 'pip install dropblock'

'''
from torchvision.models.resnet import BasicBlock,Bottleneck
from dropblock import DropBlock2D, LinearScheduler #Try to use the dropblock lib
import torch
import torch.nn as nn
import math

class ResNet(nn.Module):
    def __init__(self, cfg, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) #150*150
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.cfg = cfg
        #----------new structure called DropBlock 11sr April-------------------------
        if self.cfg.MODEL.BACKBONE.DROP_BLOCK:
            drop_prob=0.5
            block_size=3
            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_prob, block_size=block_size),
                start_value=0.,
                stop_value=drop_prob,
                nr_steps=5
            )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) #75*75
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) #38*38
        #2. extra_layers (ReLU will be used in the foward function) 25thApril,Xiaoyu Zhu
        if self.cfg.MODEL.BACKBONE.DEPTH< 50:
            self.ex_layer00 = self._make_layer(block, 512,2 , stride=2) #19*19
            self.ex_layer0 = self._make_layer(block, 256,2 , stride=2)  #10*10
            self.ex_layer1 = self._make_layer(block, 256, 2, stride=2)  #5*5
            self.ex_layer2 = self._make_layer(block, 128, 2, stride=2)  #3*3
            self.ex_layer3 = self._make_layer(block, 128, 2, stride=2)  #1*2
        else: #This if command just for reducing the amount of params to satisfy the memory of my GPU
            self.ex_layer00 = self._make_layer(block, 512,1 , stride=2) #19*19
            self.ex_layer0 = self._make_layer(block, 256,1 , stride=2)  #10*10
            self.ex_layer1 = self._make_layer(block, 256, 1, stride=2)  #5*5
            self.ex_layer2 = self._make_layer(block, 128, 1, stride=2)  #3*3
            self.ex_layer3 = self._make_layer(block, 128, 1, stride=2)  #1*2
 
        # kaiming weight normal after default initialization 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # construct layer/stage conv2_x,conv3_x,conv4_x,conv5_x
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # when to need downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # inplanes expand for next block
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def _make_extra_layers(self, input_channels, output_channels,k,s, p): #10thApril,Xiaoyu Zhu
        layers = []
        layers.append(torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels,kernel_size=k,stride=s,padding=p))
        layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.cfg.MODEL.BACKBONE.DROP_BLOCK:
            self.dropblock.step()  # increment number of iterations
        out_features = []
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu(x)
        if self.cfg.MODEL.BACKBONE.DROP_BLOCK:
            x = self.dropblock(self.layer1(x)) #added 11st April
        else:
            x = self.layer1(x)  
        if self.cfg.MODEL.BACKBONE.DROP_BLOCK:
            x = self.dropblock(self.layer2(x))
        else:
            x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)     #30*40/38*38 output[0] 
        out_features.append(x)
        #For Extra Layer: 10thApril,Xiaoyu Zhu
        x = self.ex_layer00(x) #15*20/19*19 output[1]
        out_features.append(x)
        x = self.ex_layer0(x)  #8*10/10*10 output[2]
        out_features.append(x)
        x = self.ex_layer1(x)  #4*5/5*5 output[3]
        out_features.append(x) 
        x = self.ex_layer2(x)  #2*3/3*3 output[4] 
        out_features.append(x) 
        x = self.ex_layer3(x)  #1*2/1*1 output[5] 
        out_features.append(x) 
     
        return tuple(out_features)
