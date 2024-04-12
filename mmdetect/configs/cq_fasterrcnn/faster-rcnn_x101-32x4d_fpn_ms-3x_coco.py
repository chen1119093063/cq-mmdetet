_base_ = ['../common/exdark.py', '../_base_/models/faster-rcnn_r50_fpn.py']
load_from='https://cdn-model.openxlab.org.cn/models%2Fweight%2Fmmdetection%2FFaster+R-CNN%2Ffaster_rcnn_x101_32x4d_fpn_mstrain_3x_coco_20210524_124151-16b9b260.pth?Expires=1712554855&OSSAccessKeyId=LTAI5tCdKkrGqdpR7PDyejq7&Signature=RJuKA8v6mYH1NwMd%2F72SCq%2BzGPE%3D&response-content-disposition=attachment%3B%20filename%3Dfaster_rcnn_x101_32x4d_fpn_mstrain_3x_coco_20210524_124151-16b9b260.pth'
model = dict(
    
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=12,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))
