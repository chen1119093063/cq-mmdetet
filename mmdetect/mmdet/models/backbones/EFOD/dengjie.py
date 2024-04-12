import torch
import numpy as np
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import os
import cv2
import math
import torchvision
from PIL import Image
from .odconv.od import BasicBlock
import numpy as np
from ...builder import BACKBONES

class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size=3,stride=1, padding=1,  use_bias=True, activation=nn.ReLU, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class SEAttention(nn.Module):

    def __init__(self, channel=8,reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Short Cut Connection on Final Layer
class FEN(nn.Module):
    def __init__(self):	
        super(FEN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 24
        self.e_conv0 = nn.Conv2d(3,number_f,3,1,1,bias=True)
        self.e_conv1 = BasicBlock(number_f,number_f) 
        self.e_conv2 = BasicBlock(number_f,number_f) 
        self.e_conv3 = BasicBlock(number_f,number_f) 
        self.e_conv4 = BasicBlock(number_f,number_f) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.attention_blocks = SEAttention()
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)


		
    def forward(self, x):

        x0 = self.relu(self.e_conv0(x))
        
        x1 = self.e_conv1(x0)
        # p1 = self.maxpool(x1)
        x2 = self.e_conv2(x1)
        # p2 = self.maxpool(x2)
        x3 = self.e_conv3(x2)
        # p3 = self.maxpool(x3)
        x4 = self.e_conv4(x3)

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        F = self.relu(self.e_conv7(torch.cat([x1,x6],1)))
        # initial convolution
        
        return F
        # return torch.cat([Fh1,Fh2,Fh3],1)



            
@BACKBONES.register_module()
class DJ(nn.Module):
    def __init__(self, in_dim=3):
        super(DJ, self).__init__()
        self.fen_net = FEN()
        self.feh_conv3 = ConvBlock(24,in_dim,3,1,1) 
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, I):
        B,C,H,W = I.shape
        F = self.fen_net(I)
        Hierarchical_Representation = self.feh_conv3(F)
        img_high = I + Hierarchical_Representation
    
        return  Hierarchical_Representation,img_high


