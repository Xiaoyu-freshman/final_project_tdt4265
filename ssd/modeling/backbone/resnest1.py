'''

The comments are created by Xiaoyu Zhu at 26 April.
*This backbone code is designed by Xiaoyu Zhu for TDT4265 final project. But finally this was not used, since the parameters are too much for this project as well as the training time is too long.
*with the referencing of :
1. Torch official resnet code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. ssjatmhmy's Resnet+SSD code (This is unfinished code): https://github.com/ssjatmhmy/pytorch-resnet-ssd/blob/master/resnet_base.py

*Functions:
1. _make_layer(): borrowed from Torch official resnet code.
2. _make_extra_layers(): designed by Xiaoyu, which is the naive double-layer Conv2d. Not used in the final structure.
3. kaiming weight normal borrowed from ssjatmhmy's Resnet+SSD code
*Additional Support:
1. Added the support of DropBlock, with referencing of miguelvr/dropblock (https://github.com/miguelvr/dropblock/blob/master/examples/resnet-cifar10.py)
*To install it: 'pip install dropblock'from torchvision.models.resnet import BasicBlock,Bottleneck
2. Added the support for ResNest based on Github:zhanghang1989/ResNeSt(https://github.com/zhanghang1989/ResNeSt)
*To install it: 'pip install resnest --pre'

'''
from dropblock import DropBlock2D, LinearScheduler #Try to use the dropblock lib
import torch
import torch.nn as nn
import math
from resnest.torch import resnest50


class ResNest(nn.Module):
    def __init__(self, cfg, block):
        self.inplanes = 512
        super(ResNest, self).__init__()
        self.cfg = cfg
        net = resnest50(pretrained=True)
        self.before_maxpool = nn.Sequential(*list(net.children())[:3]) 
        self.to_layer3 = nn.Sequential(*list(net.children())[4:-4])    #75*75
        self.ex_layer000 = self._make_layer(block, 1024, 1 , stride=2) #38*38
        self.ex_layer00 = self._make_layer(block, 512, 1 , stride=2)   #19*19
        self.ex_layer0 = self._make_layer(block, 512, 1 , stride=2)    #10*10
        self.ex_layer1 = self._make_layer(block, 512, 1, stride=2)     #5*5
        self.ex_layer2 = self._make_layer(block, 256, 1, stride=2)     #3*3
        self.ex_layer3 = self._make_layer(block, 128, 1, stride=2)
        

        # kaiming weight normal after default initialization
        ttt=0
        for m in self.modules():
            ttt+=1 #Control for not initializing the pre-trained parmeters.
            
            if ttt>114: #198 for 3 pre-layers
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
    def _make_extra_layers(self, input_channels, output_channels,k, p): #10thApril,Xiaoyu Zhu
        layers = []
        layers.append(torch.nn.Conv2d(in_channels=input_channels, out_channels=output_channels,kernel_size=k,stride=1,padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(output_channels))
        layers.append(torch.nn.Conv2d(in_channels=output_channels, out_channels=output_channels,kernel_size=k,stride=2,padding=p)) 
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(output_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out_features = []
        x = self.before_maxpool(x)
        x = self.to_layer3(x) #30*40/38*38 output[0]
        x = self.ex_layer000(x)
        out_features.append(x)
        x = self.ex_layer00(x) #15*20/19*19 output[1]
        out_features.append(x)
        x = self.ex_layer0(x)  #8*10/10*10 output[2]
        out_features.append(x) 
        #For Extra Layer: 10thApril, Xiaoyu Zhu
        x = self.ex_layer1(x)  #4*5/5*5 output[3] 
        out_features.append(x) 
        x = self.ex_layer2(x)  #2*3/3*2 output[4]
        out_features.append(x) 
        x = self.ex_layer3(x)  #1*2/1*1 output[5]
        out_features.append(x) 
   
        return tuple(out_features)
