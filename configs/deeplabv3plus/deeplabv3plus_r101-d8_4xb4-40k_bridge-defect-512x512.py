_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/bridge_defect.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        num_classes=5,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0, 20.2, 5.4, 9.2, 17.4])),
    auxiliary_head=dict(
        num_classes=5,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=[1.0, 20.2, 5.4, 9.2, 17.4])))

# Evaluator: IoU + TPR/FPR
val_evaluator = dict(type='TPRFPRMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator

# Visualization: save all val predictions
default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))
