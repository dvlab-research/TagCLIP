_base_ = [
    '../../_base_/models/proposed_model.py', '../../_base_/datasets/context_60_512x512.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_40k.py'
]

base_class = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 
              10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
              20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
              30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
              40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
              50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
novel_class= []
both_class = base_class
num_classes = len(both_class)
in_channels = 512

data = dict(samples_per_gpu=2,
            workers_per_gpu=4,)

model = dict(
    backbone=dict(
        type='VPTCLIPVisionTransformer',
        num_tokens=100,
    ),
    decode_head=dict(
        type='ProposedHead',
        eval_disc_weight=False, 
        discriminate=False,
        seen_idx=base_class,
        all_idx=both_class,
        num_classes=num_classes,
        embed_dims=in_channels,
        loss_decode=dict(
            type='SegLossPlus', 
            num_classes=num_classes),
    ),
    base_class = base_class,
    novel_class = novel_class,
    both_class = both_class,
    self_training = True,
    load_text_embedding='src/configs/_base_/datasets/text_embedding/context_multi.npy'
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
