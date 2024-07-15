_base_ = [
    '../_base_/models/proposed_model.py', '../_base_/datasets/voc12_20_aug_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
novel_class = [15, 16, 17, 18, 19]
both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
num_classes = len(base_class)

data = dict(samples_per_gpu=2,
            workers_per_gpu=4,)

model = dict(
    backbone=dict(
        type='VPTCLIPVisionTransformer',
        num_tokens=10,
    ),
    decode_head=dict(
        type='ProposedHead',
        eval_disc_weight=False, 
        discriminate=True,
        seen_idx=base_class,
        all_idx=both_class,
        num_classes=num_classes,
        loss_decode=dict(
            type='SegLossPlus', 
            num_classes=num_classes, 
            disc_weight=10.0
        ),
    ),
    base_class = base_class,
    novel_class = novel_class,
    both_class = both_class,
    load_text_embedding='src/configs/_base_/datasets/text_embedding/voc12_single.npy'
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)

optimizer = dict(type='AdamW', lr=0.00002, weight_decay=0.01, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=10.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.),
                                        'ln': dict(decay_mult=0.),
                                        'head': dict(lr_mult=10.),
                                        }))
