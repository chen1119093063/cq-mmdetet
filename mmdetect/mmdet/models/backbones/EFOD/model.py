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

class FC(nn.Module):
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.fc(x)
        # if self.bn:
        #     x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class SEAttention(nn.Module):

    def __init__(self, channel=32,reduction=8):
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

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):
        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)
        

class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=params.batch_norm)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Coeffs(nn.Module):

    def __init__(self, nin=3, nout=4, params=None):
        super(Coeffs, self).__init__()
        self.params = params
        self.nin = nin 
        self.nout = nout
        
        lb = params.luma_bins
        cm = params.channel_multiplier
        sb = params.spatial_bin
        bn = params.batch_norm
        nsize = params.low_size

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize/sb))
        self.splat_features = nn.ModuleList()
        prev_ch = nin
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(ConvBlock(prev_ch, cm*(2**i)*lb, 3, stride=2, batch_norm=use_bn))
            prev_ch = splat_ch = cm*(2**i)*lb

        # global features
        n_layers_global = int(np.log2(sb/4))
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        for i in range(n_layers_global):
            self.global_features_conv.append(ConvBlock(prev_ch, cm*8*lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm*8*lb

        n_total = n_layers_splat + n_layers_global
        prev_ch = prev_ch * (nsize/2**n_total)**2
        self.global_features_fc.append(FC(prev_ch, 32*cm*lb, batch_norm=bn))
        self.global_features_fc.append(FC(32*cm*lb, 16*cm*lb, batch_norm=bn))
        self.global_features_fc.append(FC(16*cm*lb, 8*cm*lb, activation=None, batch_norm=bn))

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(splat_ch, 8*cm*lb, 3, batch_norm=bn))
        self.local_features.append(ConvBlock(8*cm*lb, 8*cm*lb, 3, activation=None, use_bias=False))
        self.se_att1 = SEAttention()
        self.se_att2 = SEAttention()
        self.se_att3 = SEAttention()
        self.se_att4 = SEAttention()
        # predicton
        self.conv_out = ConvBlock(8*cm*lb, lb*nout*nin, 1, padding=0, activation=None)

   
    def forward(self, lowres_input):
        params = self.params
        bs = lowres_input.shape[0]
        lb = params.luma_bins
        cm = params.channel_multiplier
        sb = params.spatial_bin

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)
        splat_features = x
        
        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)        
        # x1 = self.se_att1(x[:, :16, :, :])
        x2= self.se_att2(x[:, 0:32, :, :])
        x3 = self.se_att3(x[:, 32:, :, :])
        # x4 = self.se_att4(x[:, 48:64, :, :])
        # x = torch.cat([x1,x2,x3,x4],dim=1)
        x = torch.cat([x2,x3],dim=1)
        local_features = x

        fusion_grid = local_features
        fusion_global = global_features.view(bs,8*cm*lb,1,1)
        fusion = self.relu( fusion_grid + fusion_global )

        x = self.conv_out(fusion)
        s = x.shape
        x = x.view(bs,self.nin*self.nout,lb,sb,sb) # B x Coefs x Luma x Spatial x Spatial
        return x

@BACKBONES.register_module()
class EFOD(nn.Module):

    def __init__(self):
        super(EFOD, self).__init__()
        opt = lambda: None
    # hdrnet 参数
        opt.luma_bins = 8
        opt.channel_multiplier = 1
        opt.spatial_bin = 8
        opt.batch_norm = True
        opt.low_size = 256
        self.coeffs = Coeffs(params=opt)
        self.guide = GuideNN(params=opt)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()
        # self.init_weights()
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
    def forward(self, fullres):
        lowres = F.interpolate(fullres, size=(256,256), mode='bilinear', align_corners=False)
        coeffs = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)
        high =  0.7 * out + 0.3 * fullres
        # high = fullres
        return out,high


#########################################################################################################
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # img = Image.open("siaosile.jpg")
    img = cv2.imread("siaosile.jpg")
    img = cv2.resize(img,(512,544))
    lowres = cv2.resize(img,(256,256))
    img = (np.asarray(img)/255.0)
    img = torch.from_numpy(img).float().permute(2,0,1)
    img = img.cuda().unsqueeze(0)
    lowres = (np.asarray(lowres)/255.0)
    lowres = torch.from_numpy(lowres).float().permute(2,0,1)
    lowres = lowres.cuda().unsqueeze(0)
    # img = cv2.resize(img,(512,512))
    # img = torch.Tensor(8, 3, 320, 352).cuda()
    opt = lambda: None
# hdrnet 参数
    opt.luma_bins = 8
    opt.channel_multiplier = 1
    opt.spatial_bin = 8
    opt.batch_norm = True
    opt.low_size = 256
    net = EFOD(opt).cuda()
    # net.init_weights()
    print('total parameters:', sum(param.numel() for param in net.parameters()))
    Hierarchical_Representation, high = net(lowres,img)
    torchvision.utils.save_image(Hierarchical_Representation, "Hierarchical_Repre.png")
    torchvision.utils.save_image(high, "high.png")
#########################################################################################################

    