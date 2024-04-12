import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
from .re_model import RetinexFormer
from mmdet.registry import MODELS
from torch.nn.parallel import DistributedDataParallel as DDP


@MODELS.register_module()
class RetinexFormerNET(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.netG = RetinexFormer(in_channels=opt['in_channels'],out_channels=opt['out_channels'],n_feat=opt['n_feat'],stage=opt['stage'],num_blocks=opt['num_blocks']).cuda()
        if opt['resume_state'] is not None:
            self.netG.load_state_dict(torch.load(opt['resume_state'])['params'], strict=True) # 
        # self.netG = self.netG.cpu
    def forward(self, x):
        # h, w = x.shape[2], x.shape[3]
        
        out = self.netG(x)
        # restored = out[:, :, :h, :w]

        # restored = torch.clamp(restored, 0, 1)

        return out