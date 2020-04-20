from torchvision.models.resnet import BasicBlock,Bottleneck
from dropblock import DropBlock2D, LinearScheduler #Try to use the dropblock lib
import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, kernel, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride,
                     padding=padding, groups=groups, bias=False)

class BasicBlock_modified(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None, groups=1,
                 base_width=64, padding=1, dilation=1, norm_layer=None):
        super(BasicBlock_modified, self).__init__()
        print(inplanes,planes)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, kernel)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel, stride,padding=padding)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




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
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #75*75
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) #38*38
        #2. extra_layers (ReLU will be used in the foward function) 10thApril,Xiaoyu Zhu
        #self.ex_conv2 = self._make_extra_layers(256,256,3,2,1) 
        self.ex_layer00 = self._make_layer(block, 512,2 , stride=2) #19*19
        self.ex_layer0 = self._make_layer(block, 256,2 , stride=2)         #10*10
        self.ex_layer1 = self._make_layer(block, 256, 2, stride=2)#5*5
        self.ex_layer2 = self._make_layer(block, 128, 2, stride=2) #3*3
        self.ex_layer3 = self._make_layer(block, 128, 2, stride=2) #1*2
        #self.ex_conv3 = self._make_extra_layers(128,64,[1,2],1,0)
        

        
        

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
        #print('Input',x.shape) #Original 300*300; For rectange input size :320*240
        x = self.conv1(x) 
        #print('self.conv1',x.shape) #150*150; 160*120
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x) 
        #print('self.maxpool',x.shape) #75*75; 80*60
        if self.cfg.MODEL.BACKBONE.DROP_BLOCK:
            x = self.dropblock(self.layer1(x))#added 11st April
        else:
            x = self.layer1(x)  
        #print('layer1',x.shape) #80*60
        #out_features.append(x) #Add new feature map 60*80
        if self.cfg.MODEL.BACKBONE.DROP_BLOCK:
            x = self.dropblock(self.layer2(x))
        else:
            x = self.layer2(x)
        x = self.layer3(x)
        #x = self.ex_conv2(x)
        #print('self.ex_conv2(x)',x.shape)  #38*38 output[0]; 30*40
        out_features.append(x)
        
        x = self.ex_layer00(x) #19*19 output[1]; 15*20
        #print('self.ex_layer00(x)',x.shape)
        out_features.append(x)
        x = self.ex_layer0(x)  
        #print('self.ex_layer0',x.shape) #10*10 output[2]; 8*10
        out_features.append(x)
        #For other output: 10thApril,Xiaoyu Zhu
        x = self.ex_layer1(x) 
        #print('ex_layer1',x.shape) #5*5 output[3] ;4*5
        out_features.append(x) 
        x = self.ex_layer2(x) 
        #print('ex_layer2',x.shape) #3*2 output[4] ;2*3
        out_features.append(x) 
        x = self.ex_layer3(x)  
        #x = self.ex_conv3(x)
        #print('self.ex_conv3',x.shape) #1*1 output[5] 
        out_features.append(x) 
#-----------------------------------Old Version-------------------        
#         #For other outputs:
#         for i, f in enumerate(self.extra_layer): #i means the index and f means the function of the layer
#             if i== len(self.extra_layer):
#                 x= f(x)
#             else:
#                 x= torch.nn.functional.relu(f(x),inplace=True)
#             if i % 2 == 1:
#                 out_features.append(x)
#-----------------------------------------------------------------      
        return tuple(out_features)
