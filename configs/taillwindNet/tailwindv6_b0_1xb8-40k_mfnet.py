_base_ = [
    '../_base_/datasets/mfnet.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

randomness = dict(seed=42, diff_rank_seed=False)

custom_imports = dict(
    imports=['mmseg_plugin', 'custom_transforms', 'custom_metrics', 'custom_modules'],
    allow_failed_imports=False,
)
num_classes = 9

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    bgr_to_rgb=False,
)


model = dict(
    type='TailwindEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='TailwindV6',
        model_name='B0',
        in_channels=4,
        rgb_channels=3,
        nir_channels=1,
        drop_path_rate=0.05,
        fusion_heads=1,
        fusion_dropout=0.05,
        fusion_sr_ratio=2,
        fusion_alignment_weight=0.1, 
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
        num_classes=9,  # MFNetDataset 有 9 个类别
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=dict(
        type='SegformerHead',
        in_channels=[160],  # 第三阶段特征
        in_index=[2],
        channels=128,
        dropout_ratio=0.1,
        num_classes=9,  # 与 MFNetDataset 类别数一致
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
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

# 评估同时记录像素准确率（aAcc=PA）以及宏观 Precision/Recall（mPrecision/mRecall）
val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore']),
    dict(type='MFNetMetric'),
]
test_evaluator = val_evaluator
