_base_ = [
    './yolov3_exdark_ours_baseline.py'
]
# model settings
train_dataloader = dict(dataset=dict(data_prefix=dict(img='JPEGImages/IMGS_KinD')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='JPEGImages/IMGS_KinD')))
test_dataloader = val_dataloader

    