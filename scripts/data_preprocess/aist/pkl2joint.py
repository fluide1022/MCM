import argparse
import glob
import os
import sys

import numpy as np
import torch
from smplx import SMPLH, SMPL
from tqdm import tqdm

sys.path.append(os.curdir)


def pkl2npy(pkl_path: str, npy_path: str):
    if os.path.basename(pkl_path).split('_')[2] == 'sBM':
        gender = 'male'
    else:
        gender = 'female'

    pkl = np.load(pkl_path, allow_pickle=True)
    smpl_poses, smpl_scaling, smpl_trans = pkl['smpl_poses'], pkl['smpl_scaling'], pkl['smpl_trans']
    t = smpl_poses.shape[0]
    if gender == 'male':
        smpl = SMPL(model_path='smpl_models/smpl/SMPLH_MALE.pkl', gender='male', batch_size=t)
    else:
        smpl = SMPL(model_path='smpl_models/smplh/SMPLH_FEMALE.pkl', gender='female', batch_size=t)

    smpl_poses, smpl_scaling, smpl_trans = pkl['smpl_poses'], pkl['smpl_scaling'], pkl['smpl_trans']
    smpl_poses = np.asarray(smpl_poses)
    smpl_scaling = np.asarray(smpl_scaling)
    smpl_trans = np.asarray(smpl_trans)
    keypoints3d = smpl.forward(
        global_orient=torch.from_numpy(smpl_poses[:, :3]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 3:66]).float(),
        transl=torch.from_numpy(smpl_trans / smpl_scaling).float(),
    ).joints.detach().numpy()[:, :22]
    np.save(npy_path, keypoints3d)
    return


if __name__ == '__main__':
    args = argparse.ArgumentParser('convert aist smpl pkl files to joint positions')
    args.add_argument('--pkl_root', type=str, default='data/aist_plusplus_final/pkl')
    args.add_argument('--joints_root', type=str, default='data/aist_plusplus_final/joints_22_fps60')
    args = args.parse_args()

    os.makedirs(args.joints_root, exist_ok=True)

    for pkl_path in tqdm(glob.glob(os.path.join(args.pkl_root, '*.pkl'))):
        filename = os.path.basename(pkl_path).split('.')[0]
        save_path = os.path.join(args.joints_root, filename + '.npy')
        pkl2npy(pkl_path, save_path)
