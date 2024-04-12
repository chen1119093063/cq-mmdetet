import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules

class networks():
    
    def __init__(self, opt):
        super().__init__()
        
    


    ####################
    # define network
    ####################


    # Generator
    def define_G(opt):
        model_opt = opt
        # print(model_opt['which_model_G'])
        # if model_opt['which_model_G'] == 'ddpm':
        #     from .ddpm_modules import diffusion, unet, diffusion_Pt
        from .ddpm_modules import diffusion, unet, diffusion_Pt
        if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
            model_opt['unet']['norm_groups']=32
        model = unet.UNet(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            norm_groups=model_opt['unet']['norm_groups'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            attn_res=model_opt['unet']['attn_res'],
            res_blocks=model_opt['unet']['res_blocks'],
            dropout=model_opt['unet']['dropout'],
            image_size=model_opt['diffusion']['image_size']
        )
        # if opt['uncertainty_train']:
        #     netG = diffusion_Pt.GaussianDiffusion(
        #         model,
        #         image_size=model_opt['diffusion']['image_size'],
        #         channels=model_opt['diffusion']['channels'],
        #         loss_type='l1',   
        #         conditional=model_opt['diffusion']['conditional'],
        #         schedule_opt=model_opt['beta_schedule']['train']
        #     )
        # else:
        netG = diffusion.GaussianDiffusion(
            model,
            image_size=model_opt['diffusion']['image_size'],
            channels=model_opt['diffusion']['channels'],
            loss_type='l1',   
            conditional=model_opt['diffusion']['conditional'],
            # schedule_opt=model_opt['beta_schedule']['train']
        )
        # if opt['phase'] == 'train':
        #     # init_weights(netG, init_type='kaiming', scale=0.1)
        #     init_weights(netG, init_type='orthogonal')
        # if opt['gpu_ids'] and opt['distributed']:
        #     assert torch.cuda.is_available()
        #     netG = nn.DataParallel(netG)
        return netG

        
