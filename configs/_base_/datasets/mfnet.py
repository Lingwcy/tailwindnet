dataset_type = 'MFNetDataset'
data_root = r"D:\dp\mmseg\data\MFNet_dataset"

# 确保自定义数据增强（如 MFNetRandomEdgeCrop）被注册到 TRANSFORMS
custom_imports = dict(imports=['custom_transforms'], allow_failed_imports=False)

img_scale = (640, 480)
crop_size = (480, 640)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged', imdecode_backend='pillow'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MFNetRandomEdgeCrop', crop_rate=0.1, prob=1.0),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged', imdecode_backend='pillow'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.txt',
        data_prefix=dict(img_path='images', seg_map_path='labels'),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type='MFNetMetric')
test_evaluator = val_evaluator
