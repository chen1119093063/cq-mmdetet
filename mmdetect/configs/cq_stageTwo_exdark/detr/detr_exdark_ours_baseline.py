_base_ = [
    './detr_exdark_baseline.py'
]

model = dict(
    type='CQ_DETR',
    pre_encoder = dict(type='CQ_new_DENet'))
    
