import argparse
import os

import kornia
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from mmdet.registry import MODELS
from .models.networks import IAN,ANSN,FuseNet


@MODELS.register_module()
class ModelBreadNet(nn.Module):
    def __init__(self,model1, model2, model3,model4,opt):
        super().__init__()
        self.eps = 1e-6
        self.model_ianet = model1(in_channels=1, out_channels=1).cuda()
        self.model_nsnet = model2(in_channels=2, out_channels=1).cuda()
        self.model_canet = model3(in_channels=4, out_channels=2).cuda()
        self.model_fdnet = model4(in_channels=3, out_channels=1).cuda()
        if opt.model1_weight is not None:
            self.load_weight(self.model_ianet, opt.model1_weight)
            self.load_weight(self.model_nsnet, opt.model2_weight)
            self.load_weight(self.model_canet, opt.model3_weight)
            self.load_weight(self.model_fdnet, opt.model4_weight)
        

    def load_weight(self, model, weight_pth):
        if model is not None:
            state_dict = torch.load(weight_pth)
            model.load_state_dict(state_dict, strict=True)

    def noise_syn_exp(self, illumi, strength):
        return torch.exp(-illumi) * strength

    def forward(self, image):
        # Color space mapping
        texture_in, cb_in, cr_in = torch.split(kornia.color.rgb_to_ycbcr(image), 1, dim=1)

        # Illumination prediction
        texture_in_down = F.interpolate(texture_in, scale_factor=0.5, mode='bicubic', align_corners=True)
        texture_illumi = self.model_ianet(texture_in_down)
        texture_illumi = F.interpolate(texture_illumi, scale_factor=2, mode='bicubic', align_corners=True)

        # Illumination adjustment
        texture_illumi = torch.clamp(texture_illumi, 0., 1.)
        texture_ia = texture_in / torch.clamp_min(texture_illumi, self.eps)
        texture_ia = torch.clamp(texture_ia, 0., 1.)

        # Noise suppression and fusion
        texture_nss = []
        for strength in [0., 0.05, 0.1]:
            attention = self.noise_syn_exp(texture_illumi, strength=strength)
            texture_res = self.model_nsnet(torch.cat([texture_ia, attention], dim=1))
            texture_ns = texture_ia + texture_res
            texture_nss.append(texture_ns)
        texture_nss = torch.cat(texture_nss, dim=1).detach()
        texture_fd = self.model_fdnet(texture_nss)

        # Further preserve the texture under brighter illumination
        texture_fd = texture_illumi * texture_in + (1 - texture_illumi) * texture_fd
        texture_fd = torch.clamp(texture_fd, 0, 1)

        # Color adaption
        colors = self.model_canet(
                torch.cat([texture_in, cb_in, cr_in, texture_fd], dim=1))
        cb_out, cr_out = torch.split(colors, 1, dim=1)
        cb_out = torch.clamp(cb_out, 0, 1)
        cr_out = torch.clamp(cr_out, 0, 1)

        # Color space mapping
        image_out = kornia.color.ycbcr_to_rgb(
            torch.cat([texture_fd, cb_out, cr_out], dim=1))

        # Further preserve the color under brighter illumination
        img_fusion = texture_illumi * image + (1 - texture_illumi) * image_out
        _, cb_fuse, cr_fuse = torch.split(kornia.color.rgb_to_ycbcr(img_fusion), 1, dim=1)
        image_out = kornia.color.ycbcr_to_rgb(
            torch.cat([texture_fd, cb_fuse, cr_fuse], dim=1))
        image_out = torch.clamp(image_out, 0, 1)
        # torchvision.utils.save_image(image,"image.jpg")
        # torchvision.utils.save_image(image_out,"image_out.jpg")
        # return texture_ia, texture_nss, texture_fd, image_out, texture_illumi, texture_res
        return image_out

