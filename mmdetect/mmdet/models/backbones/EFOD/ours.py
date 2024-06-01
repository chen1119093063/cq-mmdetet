import math
import os
import torchvision
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from mmdet.registry import MODELS
from inspect import isfunction
def exists(x):
    return x is not None
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        # self.noise_func = FeatureWiseAffine(
        #     time_emb_dim, dim_out, use_affine_level=False)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        # h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


# class SelfAttention(nn.Module):
#     def __init__(self, in_channel, n_head=1, norm_groups=6):
#         super().__init__()

#         self.n_head = n_head

#         self.norm = nn.GroupNorm(norm_groups, in_channel)
#         self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
#         self.out = nn.Conv2d(in_channel, in_channel, 1)

#     def forward(self, input):
#         batch, channel, height, width = input.shape
#         n_head = self.n_head
#         head_dim = channel // n_head

#         norm = self.norm(input)
#         qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
#         query, key, value = qkv.chunk(3, dim=2)  # bhdyx

#         attn = torch.einsum(
#             "bnchw, bncyx -> bnhwyx", query, key
#         ).contiguous() / math.sqrt(channel)
#         attn = attn.view(batch, n_head, height, width, -1)
#         attn = torch.softmax(attn, -1)
#         attn = attn.view(batch, n_head, height, width, height, width)

#         out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
#         out = self.out(out.view(batch, channel, height, width))

#         return out + input
class GAM_Attention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention, self).__init__()
 
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )
 
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )
 
    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()
 
        x = x * x_channel_att
 
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
 
        return out

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=6, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            # self.attn = SelfAttention(dim_out, norm_groups=norm_groups)
            self.attn = GAM_Attention(dim_out)

    def forward(self, x):
        x = self.res_block(x)
        if(self.with_attn):
            x = self.attn(x)
        return x

class FEN(nn.Module):
    def __init__(self):	
        super(FEN, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
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
class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image


@MODELS.register_module()
class CQ_DENet(nn.Module):
    def __init__(self,
                 num_high=2,
                 gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        self.fen = FEN()
        self.resnetBlocWithAttn0 = ResnetBlocWithAttn(3,3,norm_groups=3,with_attn=False)
        self.resnetBlocWithAttn1 = ResnetBlocWithAttn(6,3,norm_groups=3,with_attn=False)
        # self.resnetBlocWithAttn1_1 = ResnetBlocWithAttn(64,32,norm_groups=32,with_attn=False)
        self.resnetBlocWithAttn2 = ResnetBlocWithAttn(6,3,norm_groups=3,with_attn=False)
        # self.resnetBlocWithAttn2_2 = ResnetBlocWithAttn(64,32,norm_groups=32,with_attn=True)
        # self.resnetBlocWithAttn3 = ResnetBlocWithAttn(32,3,norm_groups=1,with_attn=False)
        # self.resnetBlocWithAttn4 = ResnetBlocWithAttn(32,3,norm_groups=1,with_attn=False)
        # self.resnetBlocWithAttn5 = ResnetBlocWithAttn(32,3,norm_groups=1,with_attn=False)
        self.down = nn.Conv2d(3, 3, 3, 2, 1)
        self.down2 = nn.Conv2d(3, 3, 3, 2, 1)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)
        torchvision.utils.save_image(pyrs[0], "0.png")
        torchvision.utils.save_image(pyrs[1], "1.png")
        torchvision.utils.save_image(pyrs[2], "2.png")
        # for i in range(len(pyrs)):
        #     pyrs[i] = self.fen(pyrs[i])
        trans_pyrs = []
        
        pyrs1 = self.down(pyrs[0])
        pyrs2 = torch.cat([pyrs1,pyrs[1]],1)
        pyrs[1] = self.resnetBlocWithAttn1(pyrs2)
        pyrs3 = self.down2(pyrs[1])
        pyrs4 = torch.cat([pyrs3,pyrs[2]],1)
        pyrs[2] = self.resnetBlocWithAttn2(pyrs4)
        # pyrs[0] = self.resnetBlocWithAttn0(pyrs[0])
        
        # pyrs[0] = self.resnetBlocWithAttn3(pyrs[0])
        # pyrs[1] = self.resnetBlocWithAttn4(pyrs[1])
        # pyrs[2] = self.resnetBlocWithAttn5(pyrs[2])
        torchvision.utils.save_image(pyrs[0], "00.png")
        torchvision.utils.save_image(pyrs[1], "11.png")
        torchvision.utils.save_image(pyrs[2], "22.png")
        for i in range(len(pyrs)):
            trans_pyrs.append(pyrs[-1 - i])

        out = self.lap_pyramid.pyramid_recons(trans_pyrs)
        # Hierarchical_Representation = out
        img_high = 0.5* x + 0.5*out
        torchvision.utils.save_image(out, "out.png")
        torchvision.utils.save_image(img_high, "img_high.png")
        return img_high
    