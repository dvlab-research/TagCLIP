_base_ = [
    '../_base_/models/proposed_model.py', '../_base_/datasets/cocostuff_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
20, 21, 22, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 
44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 
65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 
86, 87, 89, 90, 91, 92, 93, 95, 96, 97, 98,  99, 100, 101, 102, 103, 104, 105, 
106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 
123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 134, 135, 138, 139, 140, 141, 
142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 
159, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]
novel_class = [19, 23, 28, 29, 36, 51, 76, 88, 94, 112, 133, 136, 137, 157, 160]
both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 
79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 
99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 
115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 
131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 
147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 
163, 164, 165, 166, 167, 168, 169, 170]
num_classes = len(both_class)

data = dict(samples_per_gpu=1,
            workers_per_gpu=4,)

model = dict(
    backbone=dict(
        type='VPTCLIPVisionTransformer',
        num_tokens=100,
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
            num_classes=num_classes),
    ),
    base_class = base_class,
    novel_class = novel_class,
    both_class = both_class,
    self_training = True,
    load_text_embedding='src/configs/_base_/datasets/text_embedding/coco_multi.npy'
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
