_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/exdark.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    './retinanet_tta.py'
]
dataset_type = 'ExdarkDataset'
data_root = 'Exdark/'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth'
model = dict(
    type='CQ_RetinaNet',
    pre_encoder = dict(type='RetinexFormerNET',opt=dict(resume_state='./Retinexformer/LOL_v1.pth', in_channels=3,out_channels= 3,n_feat=40,stage=1,num_blocks=[1,2,2])),
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

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='main/train.txt',
        data_prefix=dict(img='JPEGImages/IMGS'),
        # ann_file='annotations/instances_train2017.json',
        # data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='main/val.txt',
        data_prefix=dict(img='JPEGImages/IMGS'),
        # ann_file='annotations/instances_train2017.json',
        # data_prefix=dict(img='train2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='ExdarkMetric',
    # ann_file=data_root + 'annotations/instances_val2017.json',
    metric='mAP'
    # format_only=False,
    # backend_args=backend_args
    )
test_evaluator = val_evaluator
