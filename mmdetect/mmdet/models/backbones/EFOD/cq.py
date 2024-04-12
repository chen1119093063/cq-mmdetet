import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from collections import OrderedDict
import os
import math
import cv2
import torchvision
from PIL import Image
from ...builder import BACKBONES

class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=False):
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

# class FC(nn.Module):
#     def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
#         super(FC, self).__init__()
#         self.fc = nn.Linear(int(inc), int(outc))
#         self.activation = activation() if activation else None
#         self.bn = nn.BatchNorm1d(outc) if batch_norm else None
        
#     def forward(self, x):
#         x = self.fc(x)
#         # if self.bn:
#         #     x = self.bn(x)
#         if self.activation:
#             x = self.activation(x)
#         return x

class SEAttention(nn.Module):

    def __init__(self, channel=4,reduction=4):
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


class Coeffs(nn.Module):

    def __init__(self, nin=3, nout=3, bn=True):
        super(Coeffs, self).__init__()
    
        self.relu = nn.ReLU()

        # splat features
        self.splat_features = nn.ModuleList()
        self.feh_conv2 = ConvBlock(16, 16, 3, 1, 2, batch_norm=True) 
        self.splat_conv3_1 = ConvBlock(nin, 8, 3, stride=2, batch_norm=bn)
        self.splat_conv3_2 = ConvBlock(8, 16, 3, stride=2, batch_norm=bn)
        self.splat_conv3_3 = ConvBlock(16, 32, 3, stride=2, batch_norm=bn)
        self.splat_conv3_4 = ConvBlock(32, 64, 3, stride=2, batch_norm=bn)
        self.splat_conv3_5 = ConvBlock(64, 128, 3, stride=2, batch_norm=bn)
        
        self.upv5 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv5_1 = ConvBlock(256, 128, 3, stride=1, padding=1)
        self.upv6 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv6_1 = ConvBlock(128, 64, 3, stride=1, padding=1)
        self.upv7 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv7_1 = ConvBlock(64, 32, 3, stride=1, padding=1)
        self.upv8 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv8_1 = ConvBlock(32, 16, 3, stride=1, padding=1)
        self.upv9 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.conv9_1 = ConvBlock(16, 8, 3, stride=1, padding=1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # global features
        self.global_conv1_1 = ConvBlock(128, 64, 1, padding=0, stride=1, batch_norm=bn)
        self.global_conv1_2 = ConvBlock(64, 32, 1, padding=0, stride=1, batch_norm=bn)
        self.global_conv1_3 = ConvBlock(32, 16, 1, padding=0, stride=1, batch_norm=bn)
    
        self.global_upconv1_1 = ConvBlock(32, 64, 1, padding=0, stride=1, batch_norm=bn)
        self.global_upconv1_2 = ConvBlock(64, 128, 1, padding=0, stride=1, batch_norm=bn)
        self.global_upconv1_3 = ConvBlock(128, 256, 1, padding=0, stride=1, batch_norm=bn)

        # local features
        self.local_conv3_1 = ConvBlock(128, 64, 3,  batch_norm=bn)
        self.local_conv3_2 = ConvBlock(64, 32, 3,  batch_norm=bn)
        self.local_conv3_3 = ConvBlock(32, 16, 3,  batch_norm=bn)
        # self.se_att1 = SEAttention()
        # self.se_att2 = SEAttention()
        # self.se_att3 = SEAttention()
        # self.se_att4 = SEAttention()
        # self.se_att5 = SEAttention()
        # self.se_att6 = SEAttention()
        #down
        self.feh_conv1 = ConvBlock(16, 16, 3, 1, 2, batch_norm=True) 
        self.feh_conv2 = ConvBlock(16, 16, 3, 1, 2, batch_norm=True) 
        
        # predicton
        self.conv_out = ConvBlock(8, 3, 3,activation=None)

   
    def forward(self, lowres_input):
        
        x = lowres_input
        x_0 = self.splat_conv3_1(x)
        x_1 = self.splat_conv3_2(x_0)
        high_feature = x_1 # 16 200 
        x_2 = self.splat_conv3_3(x_1)
        mid_feature = x_2  #32 100 100
        x_3 = self.splat_conv3_4(x_2)
        x_4 = self.splat_conv3_5(x_3)
        low_features = x_4 #128 25 25
        
        #low features
        x = self.global_conv1_1(low_features)
        x = self.global_conv1_2(x)
        x = self.global_conv1_3(x)
        low_global_features = x #16 25 25
        x = low_features
        x = self.local_conv3_1(x)
        x = self.local_conv3_2(x)
        x = self.local_conv3_3(x)
        low_local_features = x
        low_fusion = self.relu(low_local_features + low_global_features)
        
        #mid features
        x = mid_feature
        x = self.global_conv1_3(x)
        mid_global_features = x
        x = mid_feature
        x = self.local_conv3_3(x)
        mid_local_features = x
        mid_fusion = self.relu(mid_local_features + mid_global_features)
        
        #high features
        x = self.feh_conv2(high_feature)
        ###########################################------------
        
        high_fusion = self.feh_conv1(x)
        high_fusion = self.feh_conv2(high_fusion)
        mid_fusion = self.feh_conv1(mid_fusion)
        mid_fusion = self.feh_conv2(mid_fusion)
        
        #attention
        x = mid_fusion + high_fusion
        # x1 = self.se_att1(x[:,   :4 , :, :])
        # x2 = self.se_att2(x[:,  4:8 , :, :])
        # x3 = self.se_att3(x[:,  8:12, :, :])
        # x4 = self.se_att4(x[:, 12:16, :, :])
        # x = torch.cat([x1,x2,x3,x4],dim=1)# 16 25
        
        ###########################################------------
    
        fusion = torch.cat([x,low_fusion],dim=1)  # 32 25
        x = self.global_upconv1_1(fusion)
        x = self.global_upconv1_2(x)  #128 25
        # x = self.global_upconv1_3(x)  #256 25
        
        
        # x = self.upv5(x)  # 128 50
        x = self.conv5_1(torch.cat([x_4,x],dim=1))  #128 25
        
        x = self.upv6(x) # 64 50
        x = self.conv6_1(torch.cat([x_3,x],dim=1))
        
        x = self.upv7(x) # 32 100
        x = self.conv7_1(torch.cat([x_2,x],dim=1))
        
        x = self.upv8(x) # 16 200
        x = self.conv8_1(torch.cat([x_1,x],dim=1))
        
        x = self.upv9(x) # 8 400
        x = self.conv9_1(torch.cat([x_0,x],dim=1)) #8 400
        
        x = self.upsample(x) # 8 800

        x = self.conv_out(x)
        # s = x.shape
        # x = x.view(bs,self.nin*self.nout,lb,sb,sb) # B x Coefs x Luma x Spatial x Spatial
        return x
@BACKBONES.register_module()
class CQ(nn.Module):

    def __init__(self):
        super(CQ, self).__init__()
        self.coeffs = Coeffs()
        # self.guide = GuideNN()
        # self.slice = Slice()
        # self.cnn = ConvBlock(12, 3, 3, batch_norm=True)
        # self.apply_coeffs = ApplyCoeffs()
        # self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m   , nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:      
                    init.constant_(m.bias, 0)
    def forward(self,fullres):
        
        coeffs = self.coeffs(fullres)
        torchvision.utils.save_image(coeffs, "out.png")
        high =  0.5* coeffs + 0.5*fullres
        return coeffs,high
