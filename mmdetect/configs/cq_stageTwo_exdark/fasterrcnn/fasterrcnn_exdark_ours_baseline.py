_base_ = ['./fasterrcnn_exdark_baseline.py']
model = dict(
    type='CQ_FasterRCNN',
    pre_encoder = dict(type='CQ_new_DENet')
    )