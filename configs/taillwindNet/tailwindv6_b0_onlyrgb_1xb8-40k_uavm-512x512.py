_base_ = [
    '../_base_/datasets/uavm.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

custom_imports = dict(
    imports=['custom_transforms'],
    allow_failed_imports=False,
)

crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)

randomness = dict(seed=42, diff_rank_seed=False)

# 注意：数据集仍然提供 13 通道输入，但 RGB-only 实 nm,  验中
# 网络只使用前 3 个通道并关闭 NIR 分支和融合。

train_pipeline = [
    dict(type='LoadImageFromNpyFile', to_float32=True),
    dict(type='SelectUAVMChannels', indices=(0, 1, 2)),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromNpyFile', to_float32=True),
    dict(type='SelectUAVMChannels', indices=(0, 1, 2)),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[
        113.0615433476454,
        111.7263375346646,
        92.52531730207197,
    ],
    std=[
        79.04668887302327,
        76.7224329589104,
        69.22617912256263,
    ],
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
    bgr_to_rgb=False,
)

# 类别权重与主实验保持一致
# 顺序：[background, canadian_goldenrod, road, pine_tree, shrub, reed]
class_weight = [0.9, 1.1, 1.0, 1.0, 1.1, 1.2]

model = dict(
    type='TailwindEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='TailwindV6',
        model_name='B0',
        in_channels=3,      # 仅 RGB
        rgb_channels=3,
        nir_channels=0,
        fusion_enable=False,  # 不启用融合
        drop_path_rate=0.05,
        fusion_heads=1,
        fusion_dropout=0.05,
        fusion_sr_ratio=2,
        fusion_alignment_weight=0.0,
        mdta_heads=2,
        mdta_ffn_expansion=2.5,
        out_indices=(0, 1, 2, 3),
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=class_weight,
        ),
    ),
    auxiliary_head=dict(
        type='SegformerHead',
        in_channels=[160],
        in_index=[2],
        channels=128,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=class_weight,
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
