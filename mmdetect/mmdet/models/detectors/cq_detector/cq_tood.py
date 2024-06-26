# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..single_stage import SingleStageDetector
from typing import List, Tuple, Union
from torch import Tensor
import torchvision

@MODELS.register_module()
class CQ_TOOD(SingleStageDetector):
    r"""Implementation of `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of TOOD. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of TOOD. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

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
