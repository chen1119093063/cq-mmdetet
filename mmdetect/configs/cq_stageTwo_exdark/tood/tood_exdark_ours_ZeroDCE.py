_base_ = [
    './tood_exdark_ours_baseline.py'
]

train_dataloader = dict(dataset=dict(data_prefix=dict(img='JPEGImages/IMGS_ZeroDCE')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='JPEGImages/IMGS_ZeroDCE')))
test_dataloader = val_dataloader

