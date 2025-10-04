# Configuration for DINO with Dynamic Query Generation
# Based on the paper: "Dynamic Query Learning via Latent Patterns for Object Detection"

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Model configuration
model = dict(
    type='DINO',
    num_classes=80,
    num_queries=900,
    denoising_nums=100,
    
    # Dynamic Query Generation Parameters
    use_dynamic_queries=True,
    num_patterns=150,  # Number of base patterns
    gamma=0.5,         # Balance factor for quality-aware assignment
    beta=0.2,          # Weight for pattern diversity loss
    
    # Backbone
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3, 4),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    
    # Neck
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4
    ),
    
    # Position Embedding
    position_embedding=dict(
        type='sine',
        hidden_dim=256,
        normalize=True
    ),
    
    # Transformer
    transformer=dict(
        type='DINOTransformer',
        embed_dim=256,
        num_feature_levels=4,
        two_stage_num_proposals=900,
        
        # Dynamic Query Generation
        use_dynamic_queries=True,
        num_patterns=100,
        gamma=0.5,
        
        # Encoder
        encoder=dict(
            type='DINOTransformerEncoder',
            num_layers=6,
            embed_dim=256,
            d_ffn=1024,
            dropout=0.1,
            n_heads=8,
            activation=dict(type='ReLU', inplace=True),
            n_levels=4,
            n_points=4
        ),
        
        # Decoder
        decoder=dict(
            type='DINOTransformerDecoder',
            num_layers=6,
            embed_dim=256,
            d_ffn=1024,
            dropout=0.1,
            n_heads=8,
            activation=dict(type='ReLU', inplace=True),
            n_levels=4,
            n_points=4,
            num_classes=80
        )
    ),
    
    # Criterion
    criterion=dict(
        type='DINOCriterion',
        num_classes=80,
        matcher=dict(
            type='HungarianMatcher',
            cost_class=1.0,
            cost_bbox=5.0,
            cost_giou=2.0
        ),
        loss_cls=dict(
            type='FocalLoss',
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_giou=dict(type='GIoULoss', loss_weight=2.0),
        
        # Quality-aware one-to-many assignment
        use_quality_aware=True,
        gamma=0.5
    ),
    
    # Postprocessor
    postprocessor=dict(
        type='DINOPostProcess',
        num_classes=80,
        use_layers=6
    )
)

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=1
)

# Learning rate and optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# Learning rate scheduler
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]
)

# Data configuration
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[
                    [
                        dict(
                            type='Resize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                       (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                       (736, 1333), (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            keep_ratio=True
                        )
                    ],
                    [
                        dict(
                            type='Resize',
                            img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                            multiscale_mode='value',
                            keep_ratio=True
                        ),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True
                        ),
                        dict(
                            type='Resize',
                            img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                       (576, 1333), (608, 1333), (640, 1333),
                                       (672, 1333), (704, 1333), (736, 1333),
                                       (768, 1333), (800, 1333)],
                            multiscale_mode='value',
                            override=True,
                            keep_ratio=True
                        )
                    ]
                ]
            ),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]
    ),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ]
            )
        ]
    ),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ]
            )
        ]
    )
)

# Image normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# Logging and evaluation
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# Evaluation configuration
evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='bbox_mAP'
)

# Checkpoint configuration
checkpoint_config = dict(interval=1)

# Runtime configuration
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# Custom hooks for monitoring dynamic query generation
custom_hooks = [
    dict(
        type='PatternUsageHook',
        priority='NORMAL'
    )
]

# Seed for reproducibility
seed = 42
deterministic = True
