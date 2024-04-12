
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
from .model import enhance_net_nopool
from mmdet.registry import MODELS
from torch.nn.parallel import DistributedDataParallel as DDP


@MODELS.register_module()
class Zero_DCE(nn.Module):
    def __init__(self):
        super().__init__()

        self.DCE_net = enhance_net_nopool().cuda()
        self.DCE_net.load_state_dict(torch.load('./Zero-DCE/Epoch99.pth'))
        # self.netG = self.netG.cpu
    def forward(self, x):
        # h, w = x.shape[2], x.shape[3]
        _,enhanced_image,_ = self.DCE_net(x)
        
        # restored = out[:, :, :h, :w]

        # restored = torch.clamp(restored, 0, 1)

        return enhanced_image


    
    




		

