_base_ = ['./fasterrcnn_exdark_ours_baseline.py']

train_dataloader = dict(dataset=dict(data_prefix=dict(img='JPEGImages/IMGS_pairLIE')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='JPEGImages/IMGS_pairLIE')))
test_dataloader = val_dataloader

