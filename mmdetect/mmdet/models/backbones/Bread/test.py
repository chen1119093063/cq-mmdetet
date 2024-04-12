import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
from mmdet.registry import MODELS
from torch.nn.parallel import DistributedDataParallel as DDP
from .models.networks import IAN,ANSN,FuseNet
from .test_Bread import ModelBreadNet

@MODELS.register_module()
class BreadNET(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.model1  = IAN
        self.model2  = ANSN
        self.model3  = FuseNet
        self.model4  = FuseNet
        self.model = ModelBreadNet(self.model1, self.model2, self.model3,self.model4,opt)
        self.model = nn.DataParallel(self.model)
        # self.netG = self.netG.cpu
    def forward(self, x):
        # h, w = x.shape[2], x.shape[3]
        
        out = self.model(x)
        # restored = out[:, :, :h, :w]

        # restored = torch.clamp(restored, 0, 1)

        return out