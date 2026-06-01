train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
	timer=dict(type='IterTimerHook'),
	logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
	param_scheduler=dict(type='ParamSchedulerHook'),
	checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='mIoU'),
	sampler_seed=dict(type='DistSamplerSeedHook'),
)
