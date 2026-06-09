_base_ = [
    '../../_base_/models/segformer_mit-b0.py',
    '../../_base_/datasets/bridge_defect_binary.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_40k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

# Baseline, num_classes=2 for binary
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=2))

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

# Option B: 100K iter (~5.9 epoch on 33K data)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=100000, by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=10000)

# Train: full 33K efflorescence images (~5.9 epoch)
# Val: keep existing 200 images for fair comparison
train_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(data_prefix=dict(
        img_path='images/train_full/efflorescence',
        seg_map_path='binary_masks/train_full/efflorescence')))

val_dataloader = dict(
    batch_size=1, num_workers=4,
    dataset=dict(data_prefix=dict(
        img_path='images/val/efflorescence',
        seg_map_path='binary_masks/val/efflorescence')))

test_dataloader = val_dataloader

# Save checkpoint every 10K iter (10 total)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000))
