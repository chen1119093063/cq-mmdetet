_base_ = [
    './yolov3_exdark_baseline.py'
]
# model settings
model = dict(
    type='CQ_TOOD',
    pre_encoder = dict(type='CQ_new_DENet'))
    