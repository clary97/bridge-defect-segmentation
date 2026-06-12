_base_ = ['./mask2former_swin-b_40k_dacl-efflorescence-512x512.py']

# Swin-L: 더 큰 backbone (embed_dims 128 -> 192)
pretrained = 'checkpoints/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'

model = dict(
    backbone=dict(
        embed_dims=192,
        num_heads=[6, 12, 24, 48],
        with_cp=True,  # gradient checkpointing for memory saving
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(in_channels=[192, 384, 768, 1536]))

# Swin-L은 큰 모델이라 batch_size=1로 줄여서 OOM 방지
train_dataloader = dict(batch_size=1, num_workers=2)
