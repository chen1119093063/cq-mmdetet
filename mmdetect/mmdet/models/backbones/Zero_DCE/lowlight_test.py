
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
    def __init__(self,opt):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.DCE_net = enhance_net_nopool().cuda()
        if opt.model_weight is not None:
            self.DCE_net.load_state_dict(torch.load(opt.model_weight))
        # self.netG = self.netG.cpu
    def forward(self, x):
        # h, w = x.shape[2], x.shape[3]
        x = self.relu(x)
        _,enhanced_image,_ = self.DCE_net(x)
        
        # restored = out[:, :, :h, :w]

        # restored = torch.clamp(restored, 0, 1)

        return enhanced_image


    
    




		

