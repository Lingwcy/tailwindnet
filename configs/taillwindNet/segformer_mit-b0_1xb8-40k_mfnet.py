_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/mfnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
randomness = dict(seed=42, diff_rank_seed=False)

custom_imports = dict(
    imports=['custom_transforms', 'custom_metrics', 'custom_modules'],
    allow_failed_imports=False,
)

num_classes = 9

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.0, 0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0, 255.0],
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    bgr_to_rgb=False,
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=4,
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        drop_rate=0.0,
        drop_path_rate=0.05,
        qkv_bias=True,
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    loss_scale='dynamic',
)

param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=1000,
        start_factor=1e-3,
        by_epoch=False,
    ),
    dict(
        type='PolyLR',
        eta_min=1e-5,
        power=0.9,
        begin=1000,
        end=40000,
        by_epoch=False,
    ),
]

default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=False),
)

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore']),
    dict(type='MFNetMetric'),
]
test_evaluator = val_evaluator
