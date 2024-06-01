# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..single_stage import SingleStageDetector
from typing import List, Tuple, Union
from torch import Tensor
import torch
import torchvision

@MODELS.register_module()
class CQ_YOLOV3(SingleStageDetector):
    
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 pre_encoder: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.pre_encoder = MODELS.build(pre_encoder)
        # self.pre_encoder.netG.load_state_dict(torch.load('./Retinexformer/LOL_v1.pth')['params'], strict=True) # 
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
            """Extract features.

            Args:
                batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

            Returns:
                tuple[Tensor]: Multi-level features that may have
                different resolutions.
            """
            x = self.pre_encoder(batch_inputs)
            # torchvision.utils.save_image(x,"123.jpg")
            x = self.backbone(x)
            if self.with_neck:
                x = self.neck(x)
            return x
