_base_ = [
    './detr_exdark_ours_baseline.py'
]

train_dataloader = dict(dataset=dict(data_prefix=dict(img='JPEGImages/IMGS_GSAD')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='JPEGImages/IMGS_GSAD')))
test_dataloader = val_dataloader

