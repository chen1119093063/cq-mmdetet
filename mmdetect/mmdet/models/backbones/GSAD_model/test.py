import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
from .model import DDPM as M
from mmdet.registry import MODELS
from torch.nn.parallel import DistributedDataParallel as DDP


@MODELS.register_module()
class GSAD(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.diffusion = M(opt)
        
        self.diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule']['val'], schedule_phase='val')

    

        # self.netG = self.netG.cpu
    def forward(self, x):
        # h, w = x.shape[2], x.shape[3]
        device = x.device
        val_data = {}
        val_data['LQ'] = x
        val_data['GT'] = x
        self.diffusion.feed_data(val_data)
        self.diffusion.test(continous=False)

        visuals = self.diffusion.get_current_visuals()
        out = visuals['HQ'].to(device)
        
        # restored = out[:, :, :h, :w]

        # restored = torch.clamp(restored, 0, 1)

        return out