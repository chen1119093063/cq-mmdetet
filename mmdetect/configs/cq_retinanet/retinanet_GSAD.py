_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/exdark.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    './retinanet_tta.py'
]
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth'
model = dict(
    type='CQ_RetinaNet',
    # pre_encoder = dict(type='GSAD',opt=dict(resume_state='./GSAD/lolv2_syn_gen.pth',gpu_ids="0",model = dict(beta_schedule=dict(train=dict(schedule='linear',
    #             n_timestep=500,
    #             linear_start= 1e-4,
    #             linear_end=2e-2),val=dict(schedule='linear',
    #             n_timestep=10,
    #             linear_start= 2e-3,
    #             linear_end=9e-1)))
    #         ,unet = dict(in_channel= 6,out_channel=3,inner_channel=64,
    # channel_multiplier= [1,1,2,2,4],attn_res=[16],res_blocks=2,dropout=0), diffusion = dict(image_size=128,channels=6, conditional=True))),
    pre_encoder = dict(type='GSAD',opt=dict(resume_state=None,gpu_ids="0",model = dict(beta_schedule=dict(train=dict(schedule='linear',
                n_timestep=500,
                linear_start= 1e-4,
                linear_end=2e-2),val=dict(schedule='linear',
                n_timestep=10,
                linear_start= 2e-3,
                linear_end=9e-1)))
            ,unet = dict(in_channel= 6,out_channel=3,inner_channel=64,
    channel_multiplier= [1,1,2,2,4],attn_res=[16],res_blocks=2,dropout=0), diffusion = dict(image_size=128,channels=6, conditional=True))),
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
    bbox_head=dict(
        type='RetinaHead',
        num_classes=12,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001))
