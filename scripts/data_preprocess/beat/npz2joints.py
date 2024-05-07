import os
from glob import glob
from os.path import join, basename

import fire
import numpy as np
import torch
from smplx import SMPLX
from smplx.utils import SMPLXOutput
from torch import from_numpy
from tqdm import tqdm


def smplx2joints52(npz_path, save_path):
    npz = np.load(npz_path, allow_pickle=True)
    gender = str(npz['gender'])
    pose = npz['poses']
    trans = npz['trans']
    t = min(
        pose.shape[0],
        trans.shape[0]
    )
    pose = pose[:t]
    trans = trans[:t]

    global_orient = pose[:, :3]
    body_pose = pose[:, 3:66]
    left_hand = pose[:, 75:120]
    right_hand = pose[:, 120:]

    if gender == 'male':
        model = SMPLX('smpl_models/smplx/SMPLX_MALE.npz', gender='male', use_pca=False, batch_size=t).cuda()
    elif gender == 'female':
        model = SMPLX('smpl_models/smplx/SMPLX_FEMALE.npz', gender='female', use_pca=False, batch_size=t).cuda()
    else:
        model = SMPLX('smpl_models/smplx/SMPLX_NEUTRAL.npz', gender='neutral', use_pca=False, batch_size=t).cuda()

    with torch.no_grad():
        output: SMPLXOutput = model.forward(
            global_orient=from_numpy(global_orient).cuda(),
            body_pose=from_numpy(body_pose).cuda(),
            left_hand_pose=from_numpy(left_hand).cuda(),
            right_hand_pose=from_numpy(right_hand).cuda(),
            transl=from_numpy(trans).cuda()
        )
    joints = output.joints.cpu().numpy()
    joints_22 = joints[:, :22]
    assert joints_22.shape[1] == 22
    np.save(save_path, joints_22)


def main(
        npz_root: str = 'data/beat/npz',
        save_root: str = 'data/beat/joints_22_fps30'
):
    os.makedirs(save_root, exist_ok=True)
    for npz_path in tqdm(glob(join(npz_root, '*.npz'))):
        save_path = join(save_root, basename(npz_path).replace('.npz', '.npy'))
        if os.path.exists(save_path):
            continue
        smplx2joints52(npz_path, save_path)


if __name__ == '__main__':
    fire.Fire(main)
