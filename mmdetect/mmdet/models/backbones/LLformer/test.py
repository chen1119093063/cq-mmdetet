## Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method
## Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, Tong Lu
## https://arxiv.org/pdf/2212.11548.pdf

import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision
from .model import LLFormer
from mmdet.registry import MODELS
from torch.nn.parallel import DistributedDataParallel as DDP


@MODELS.register_module()
class LLformerNET(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.model = LLFormer(inp_channels=3,out_channels=3,dim = 16,num_blocks = [2,4,8,16],num_refinement_blocks = 2,heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias',attention=True,skip = False)
        self.model.cuda()
        if opt.weights is not None:
            self.load_checkpoint(opt.weights)
        
        
    def load_checkpoint(self, weights):    
        checkpoint = torch.load(weights)
        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            
    def forward(self, x):
        
        b,c,h,w = x.shape
        input = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        # torchvision.utils.save_image(x,"x.jpg")
        restored = self.model(input)
        out = F.interpolate(restored, size=(h, w), mode='bilinear', align_corners=False)
        # torchvision.utils.save_image(x,"x.jpg")
        # torchvision.utils.save_image(restored,"restored.jpg")
        # torchvision.utils.save_image(out,"out.jpg")
        return out




    




# Load corresponding models architecture and weights


