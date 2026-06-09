_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/bridge_defect.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

# model: B0 + CW + DiceLoss (Exp 4와 동일)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(
        num_classes=5,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[1.0, 20.2, 5.4, 9.2, 17.4]),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                naive_dice=True,
                loss_weight=3.0,
                ignore_index=255)
        ]))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# iter 20K로 변경 (Exp 4의 40K와 동일 에포크 유지)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0,
        end=750),  # warmup도 비례해서 절반으로
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=750,
        end=20000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)

# batch_size 2 → 4로 변경
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

# Checkpoint interval도 절반으로
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))

# Evaluator
val_evaluator = dict(type='TPRFPRMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
