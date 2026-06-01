_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/uavm.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)

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

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=13,
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
        num_classes=6,
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


# 评估同时记录像素准确率（aAcc=PA）以及宏观 Precision/Recall（mPrecision/mRecall）
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator