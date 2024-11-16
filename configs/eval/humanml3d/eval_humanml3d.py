"""
    for evaluation on text-to-motion task on humanml3d
"""

eval_cfg = dict(
    type='T2MExtractor',
    dim_pose=263,
    unit_length=4,
    dim_movement_enc_hidden=512,
    dim_movement_latent=512,
    dim_word=300,
    dim_text_hidden=512,
    dim_coemb_hidden=512,
    max_text_len=20,
    diverisity_times=300,
    mm_num_samples=100,
    mm_num_repeats=30,
    mm_num_times=10,
    glove_path='checkpoints/glove',
    eval_model='checkpoints/humanml3d/text_mot_match/model/finest.tar',
)

train_dataloader = None

test_dataloader = dict(
    type='EvalHumanml3dDataset',
    batch_size=32,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(
        type='pad_collate_function'
    ),
    eval_cfg=eval_cfg,
)
