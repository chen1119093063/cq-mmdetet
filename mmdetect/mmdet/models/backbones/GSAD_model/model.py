import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os

from .base_model import BaseModel
from .networks import networks
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)

        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.opt = opt
        # self.set_new_noise_schedule(
        #     opt['model']['beta_schedule']['train'], schedule_phase='train')
        # self.device = torch.device("cuda", self.local_rank)
        self.netG.set_new_noise_schedule(opt['model']['beta_schedule']['train'], self.device)
        if self.opt['resume_state'] is not None:
            self.netG.load_state_dict(torch.load(self.opt['resume_state']), strict=True)
        
    def feed_data(self, data):

        dic = {}

        
        dic['LQ'] = data['LQ']
        dic['GT'] = data['GT']

        self.data = self.set_device(dic)
    def test(self, continous=False):
        
        self.SR = self.netG.super_resolution(self.data['LQ'], continous)
        
    
    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):

    
        self.schedule_phase = schedule_phase
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_new_noise_schedule(
                schedule_opt, self.device)
        else:
            self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_visuals(self, need_LR=True, sample=False):
            out_dict = OrderedDict()
            if sample:
                out_dict['SAM'] = self.SR.detach().float().cpu()
            else:
                out_dict['HQ'] = self.SR.detach().float().cpu()
                out_dict['INF'] = self.data['LQ'].detach().float().cpu()
                out_dict['GT'] = self.data['GT'].detach()[0].float().cpu()
                if need_LR and 'LR' in self.data:
                    out_dict['LQ'] = self.data['LQ'].detach().float().cpu()
                else:
                    out_dict['LQ'] = out_dict['INF']
            return out_dict