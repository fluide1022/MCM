_base_ = '../_base_/default_runtime.py'
max_motion_len = 196


train = dict(
    num_epochs=2000,
    log_freq=50,  # step
    save_latest_freq=1000,  # step
    save_freq=1000  # epoch
)

eval_cfg = dict(
    type='T2MExtractor',
    task='t2m',
    dim_pose=263,
    unit_length=4,
    dim_motion_hidden=1024,
    dim_movement_enc_hidden=512,
    dim_movement_latent=512,
    dim_word=300,
    dim_text_hidden=512,
    dim_coemb_hidden=512,
    max_text_len=20,
    max_motion_len=196,
    diversity_times=300,
    mm_num_samples=100,
    mm_num_repeats=30,
    mm_num_times=10,
    glove_path='checkpoints/glove',
    eval_model='checkpoints/humanml3d/text_mot_match/model/finest.tar',
)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4, betas=[0.9, 0.99], weight_decay=0.0
    )
)
diffusion = dict(
    diffusion_steps=1000,
    pred_xstart=True
)

model = dict(
    type='MCM',
    clip_version='ViT-B/32',
    input_feats=263,
    num_frames=196,
    latent_dim=512,
    ff_size=1024,
    num_layers=8,
    num_heads=8,
    dropout=0.,
    activation="gelu",
    num_text_layers=4,
    text_latent_dim=256,
    text_ff_size=2048,
    text_num_heads=4,
    chan_first=True,
    use_control=False
)
train_dataloader = dict(
    batch_size=160,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(
        type='pad_collate_function'
    ),
    dataset=dict(
        type='MotionTextDataset',
        data_root=['data/humanml3d', 'data/aist_plusplus_final', 'data/beat'],
        motion_prefix='vecs_joints_22',
        caption_prefix='texts',
        index_file=['train_val.txt', 'train.txt', 'all.txt'],
        mean_file='data/global_mean.npy',
        std_file='data/global_std.npy',
        ignore_file='ignore.txt',
        max_motion_length=max_motion_len
    )

)

test_dataloader = dict(
    batch_size=1024,
    num_workers=0,
    # persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(
        type='sort_by_sent_len_collate_function'
    ),
    dataset=dict(
        type='EvalT2MDataset',
        eval_cfg=eval_cfg
    )
)
