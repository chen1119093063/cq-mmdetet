
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
from .models import *
from mmdet.registry import MODELS
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision

@MODELS.register_module()
class KinDNet(nn.Module):
    def __init__(self,opt):
        super().__init__()

        self.model = KinD()
        
        pretrain_decom = torch.load(opt.decom_net_dir)
        pretrain_resotre = torch.load(opt.restore_net_dir)
        pretrain_illum = torch.load(opt.illum_net_dir)
        if opt.weights is not None:
            self.model.decom_net.load_state_dict(pretrain_decom)
            
            self.model.restore_net.load_state_dict(pretrain_resotre)
            
            
            self.model.illum_net.load_state_dict(pretrain_illum)
            
        # self.netG = self.netG.cpu
    def forward(self, x):
        b,h, w =x.shape[0], x.shape[2], x.shape[3]
        input_low_eval_tensor = x
        # with torch.no_grad():
        decom_r_low, decom_i_low = self.model.decom_net(x)
        restoration_r = self.model.restore_net(decom_r_low, decom_i_low)
        ratio = 5.0
        i_low_data_ratio = torch.ones(h, w) * ratio
        i_low_ratio_expand = i_low_data_ratio.unsqueeze(2)
        i_low_ratio_expand2 = i_low_ratio_expand.unsqueeze(0)
        repeated_tensor = i_low_ratio_expand2.repeat(b, 1, 1, 1)
        transposed_tensor = repeated_tensor.permute(0, 3, 1, 2).cuda()
        
        adjust_i = self.model.illum_net(decom_i_low, transposed_tensor)
        r_gray = torch.mean(decom_r_low, dim=1, keepdim=True).cpu()
        Gauss = torch.as_tensor(
                    np.array([[0.0947416, 0.118318, 0.0947416],
                            [ 0.118318, 0.147761, 0.118318],
                            [0.0947416, 0.118318, 0.0947416]]).astype(np.float32)
                    ).to(r_gray.device)
        channels = r_gray.size()[1]
        Gauss_kernel = Gauss.expand(channels, channels, 3, 3)
        smoothed_image = torch.nn.functional.conv2d(r_gray, weight=Gauss_kernel, padding=1)
        low_i = torch.minimum((smoothed_image * 2).sqrt(), torch.tensor(1.0))
        i_low_ratio_expand2 = low_i.cuda()
        
        result_denoise = restoration_r * i_low_ratio_expand2
        fusion4 = result_denoise * adjust_i

        fusion2 = decom_i_low * input_low_eval_tensor + (1 - decom_i_low) * fusion4
        # torchvision.utils.save_image(x,"x.jpg")
        # torchvision.utils.save_image(fusion2,"fusion2.jpg")
        return fusion2
    

        
        




		

