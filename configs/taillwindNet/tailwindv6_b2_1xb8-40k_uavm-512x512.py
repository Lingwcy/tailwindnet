_base_ = [
    '../_base_/datasets/uavm.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]
custom_imports = dict(
    imports=['mmseg_plugin', 'custom_transforms'],
    allow_failed_imports=False,
)

crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)

randomness = dict(seed=42, diff_rank_seed=False)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[
        113.0615433476454,
        111.7263375346646,
        92.52531730207197,
        11.363117932980778,
        14.308951835398293,
        18.975172311981762,
        22.73015672648611,
        25.862042457486954,
        26.093917791682518,
        33.31593697553763,
        20.692559430643094,
        25.130988600795256,
        29.138572089072387,
    ],
    std=[
        79.04668887302327,
        76.7224329589104,
        69.22617912256263,
        15.890902401769514,
        20.25108187694789,
        27.528344137237877,
        32.94851308873022,
        37.43713494467441,
        37.7461713502959,
        45.54257025689651,
        33.551948741904596,
        37.68988879332381,
        41.621185331162806,
    ],
    size=crop_size,
    pad_val=0,
    seg_pad_val=255,
    bgr_to_rgb=False,
)

# 与 v5_1 实验12 保持一致的类别权重设置
# 顺序：[background, canadian_goldenrod, road, pine_tree, shrub, reed]
class_weight = [0.9, 1.1, 1.0, 1.0, 1.1, 1.2]

model = dict(
    type='TailwindEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='TailwindV6',
        model_name='B2',
        in_channels=13,
        rgb_channels=3,
        nir_channels=10,
        drop_path_rate=0.08,
        fusion_heads=2,
        fusion_dropout=0.05,
        fusion_sr_ratio=2,
        fusion_alignment_weight=0.1,
        mdta_heads=4,
        mdta_ffn_expansion=2.75,
        out_indices=(0, 1, 2, 3),
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
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
        in_channels=[320],
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
    clip_grad=dict(_delete_=True, max_norm=1.0, norm_type=2),
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
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
