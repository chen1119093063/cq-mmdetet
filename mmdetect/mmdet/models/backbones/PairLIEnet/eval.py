
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision
from .net import net
from mmdet.registry import MODELS
from torch.nn.parallel import DistributedDataParallel as DDP


@MODELS.register_module()
class pairLIENET(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.model = net().cuda()
        if opt.weights is not None:
            
            self.model.load_state_dict(torch.load(opt.weights, map_location=lambda storage, loc: storage))
        
        
            
    def forward(self, x):
        
        b,c,h,w = x.shape
        L, R, X = self.model(x)
        D = x- X        
        I = torch.pow(L,0.2) * R  # default=0.2, LOL=0.14.
        # flops, params = profile(model, (input,))
        torchvision.utils.save_image(x,"x.jpg")
        # torchvision.utils.save_image(restored,"restored.jpg")
        torchvision.utils.save_image(I,"out.jpg")
        return I


