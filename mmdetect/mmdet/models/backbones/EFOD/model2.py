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
import numpy as np
from ...builder import BACKBONES


def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                init.constant_(m.bias, 0)
            
class SEAttention(nn.Module):

    def __init__(self, channel=24,reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=3):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class SEAttention(nn.Module):

    def __init__(self, channel=3,reduction=3):
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
    
class CBAMBlock(nn.Module):

    def __init__(self, channel=3, reduction=3, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class SAFA(nn.Module):
    def __init__(self,in_dim,n):
        super(SAFA, self).__init__()
        # initial convolution
        
        self.conv1 = nn.Conv2d(in_dim,in_dim,7,4,2,bias=True) 
        self.conv2 = nn.Conv2d(in_dim,in_dim,3,2,1,bias=True) 
        self.conv3 = nn.Conv2d(in_dim,in_dim,3,2,1,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.attention_blocks = SEAttention()

    def forward(self, F, Fq):
        Q = self.conv1(F)
        Q = self.conv2(Q)
        K = self.conv3(Fq)
        Fqk = Q + K
        Fh1 = self.attention_blocks(Fqk[:,   :3 , :, :])
        Fh2 = self.attention_blocks(Fqk[:,   3:6 , :, :])
        Fh3 = self.attention_blocks(Fqk[:,   6:9 , :, :])
        Fh4 = self.attention_blocks(Fqk[:,   9:12 , :, :])
        Fh5 = self.attention_blocks(Fqk[:,   12:15 , :, :])
        Fh6 = self.attention_blocks(Fqk[:,   15:18 , :, :])
        Fh7 = self.attention_blocks(Fqk[:,   18:21 , :, :])
        Fh8 = self.attention_blocks(Fqk[:,   21:24 , :, :])
        
        return torch.cat([Fh1,Fh2,Fh3,Fh4,Fh5,Fh6,Fh7,Fh8],1)

# Short Cut Connection on Final Layer
class FEN(nn.Module):
    def __init__(self):	
        super(FEN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 24
        self.e_conv0 = nn.Conv2d(3,number_f,3,1,1,bias=True)
        self.e_conv1 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
              
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)


		
    def forward(self, x):

        x0 = self.relu(self.e_conv0(x))
        
        x1 = self.relu(self.e_conv1(x0))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        F = self.relu(self.e_conv7(torch.cat([x1,x6],1)))
        # initial convolution
        
        
        
        return F


            
@BACKBONES.register_module()
class FEH(nn.Module):
    def __init__(self, in_dim=3):
        super(FEH, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.feh_conv1 = nn.Conv2d(in_dim,in_dim,7,4,2,bias=True) 
        self.feh_conv2 = nn.Conv2d(in_dim,in_dim,3,2,1,bias=True)
        self.feh_conv3 = nn.Conv2d(48,in_dim,3,1,1,bias=True) 
        self.fen_net = FEN()
        self.safa_net = SAFA(in_dim*8,n = 8)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)
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
        Iq = self.relu(self.feh_conv1(I))
        Io = self.relu(self.feh_conv2(Iq))
        # Iq = self.avg1(I)
        # Io = self.avg2(Iq)
        F = self.fen_net(I)
        Fq = self.fen_net(Iq)
        Fo = self.fen_net(Io)
        Fh = self.safa_net(F,Fq)
        
        Fo_high = self.upsample(Fo)
        Fh_high = self.upsample(Fh)
        # final = Fo_high + Fh_high
        Hierarchical_Representation = self.relu(self.feh_conv3(torch.cat([Fo_high,Fh_high],dim=1)))
        
        img_high = 0.5 * I + 0.5* Hierarchical_Representation
    
        return  Hierarchical_Representation,img_high


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # img = Image.open("2015_00006.jpg")
    img = cv2.imread("2015_00006.jpg")
    img = cv2.resize(img,(512,512))
    # lowres = cv2.resize(img,(256,256))
    img = (np.asarray(img)/255.0)
    img = torch.from_numpy(img).float().permute(2,0,1)
    img = img.cuda().unsqueeze(0)
    # img = torch.Tensor(8, 3, 400, 600).cuda()
    net = FEH().cuda()
    # net.init_weights()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    Hierarchical_Representation,high = net(img)
    torchvision.utils.save_image(Hierarchical_Representation, "Hierarchical_Repre.png")
    torchvision.utils.save_image(high, "high.png")