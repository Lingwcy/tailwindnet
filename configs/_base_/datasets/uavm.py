dataset_type = 'UAVMDataset'
# 统一设置数据根目录，便于在 dataloader 中复用
data_root = r"D:\dp\mmseg\data\augment"

# 标签重映射：将稀有类别统一并压缩为 6 个可学习的标签
label_map = {
    0: 0,
    1: 1,
    2: 0,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
}

crop_size = (512, 512)
# UAVM 训练增强流水线，先读入 npy 影像再做尺度/裁剪/翻转增强
train_pipeline = [
    dict(type='LoadImageFromNpyFile', to_float32=False),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
# 验证/测试沿用固定 Resize，避免随机性，保持评估一致
test_pipeline = [
    dict(type='LoadImageFromNpyFile', to_float32=False),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 训练集：使用无限采样器以支持迭代式训练循环
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='imageSet/trn.txt',
        label_map=label_map,
        data_prefix=dict(
            img_path='processed',
            seg_map_path='SegmentationClass'),
        pipeline=train_pipeline))

# 验证集：默认顺序采样，指定 val.txt 分割列表
val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='imageSet/val.txt',
        label_map=label_map,
        data_prefix=dict(
            img_path='processed',
            seg_map_path='SegmentationClass'),
        pipeline=test_pipeline))

# 测试集沿用验证配置，便于直接复用评估脚本
test_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='imageSet/val.txt',
        label_map=label_map,
        data_prefix=dict(
            img_path='processed',
            seg_map_path='SegmentationClass'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
