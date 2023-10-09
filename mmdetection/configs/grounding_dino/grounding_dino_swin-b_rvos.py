_base_ = [
    './grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py',
]

custom_hooks = [
    dict(type='FreezeLayerHook')
]

model = dict(
    type='GroundingDINO',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        patch_norm=True),
    neck=dict(in_channels=[256, 512, 1024]),
    # training and testing settings
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    # test_cfg=dict(max_per_img=300)
)

load_from = 'mm_weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
#     dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'text', 'custom_entities'))
# ]

# TODO: update the dataset path
dataset_type = 'YTVOSDataset'
data_root = './data/ref-youtube-vos'

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations/instances_train2017.json',
        # data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        ytvos_dict=dict(
            img_folder=f'{data_root}/train/',
            ann_file=f'{data_root}/meta_expressions/train/meta_expressions.json',
            # transforms='',
            num_frames=1, # NOTE: should align with args. in inference?
            num_clips=1, # 1 for online
            sampler_interval=3,
            # sampler_steps=4,
            sampler_steps=[],
            # sampler_lengths=[2, 3]
            sampler_lengths=[]

        ),
        backend_args=_base_.backend_args),
)

val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

# train_dataloader = dict(
#     dataset=dict(
#         filter_cfg=dict(filter_empty_gt=False),
#         pipeline=train_pipeline,
#         return_classes=True))
# val_dataloader = dict(
#     dataset=dict(pipeline=test_pipeline, return_classes=True))
# test_dataloader = val_dataloader

# We did not adopt the official 24e optimizer strategy
# because the results indicate that the current strategy is superior.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1)
    }))



# learning policy
max_epochs = 12
train_cfg=dict(
        type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1,
        # assigner=dict(
        #     type='HungarianAssigner',
        #     match_costs=[
        #         dict(type='BinaryFocalLossCost', weight=2.0),
        #         dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
        #         dict(type='IoUCost', iou_mode='giou', weight=2.0)
        #     ]),
)
# train_cfg = dict(
#     type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# val_cfg = dict(type='ValLoop')
# test_cfg=dict(max_per_img=300) # NOTE: ???
# test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)

find_unused_parameters=True
