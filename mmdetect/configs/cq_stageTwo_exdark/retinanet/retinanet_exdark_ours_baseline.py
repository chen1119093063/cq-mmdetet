_base_ = [
    './retinanet_exdark_baseline.py'
]

model = dict(
    type='CQ_RetinaNet',
    pre_encoder = dict(type='CQ_new_DENet'))