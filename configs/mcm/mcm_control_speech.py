_base_ = 'mcm_no_control.py'

train = dict(
    num_epochs=10000
)

model = dict(
    use_control=True
)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4, betas=[0.9, 0.99], weight_decay=0.0
    )
)

train_dataloader = dict(
    batch_size=120,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(
        type='pad_collate_function'
    ),
    dataset=dict(
        type='MotionTextSoundDataset',
        data_root=['data/beat'],
        motion_prefix='vecs_joints_22',
        caption_prefix='texts',
        audio_prefix='jukebox',
        index_file=['train_additional.txt'],
        mean_file='data/global_mean.npy',
        std_file='data/global_std.npy',
        ignore_file='ignore.txt',

    )

)

test_dataloader= dict(
    batch_size=1,   # must be
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(
        type='pad_collate_function'
    ),
    dataset=dict(
        type='EvalM2DDataset',
        data_root='data/aist_plusplus_final',
        motion_prefix='vecs_joints_22',
        caption_prefix='texts',
        jukebox_prefix='jukebox',
        audio_prefix = 'raw_music',
        beat_prefix='music_beat',
        index_file='val_test.txt',
        mean_file='data/global_mean.npy',
        std_file='data/global_std.npy',
        ignore_file='ignore.txt',
        clip_length=80
    )

)
