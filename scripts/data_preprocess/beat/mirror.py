import os
import shutil

import numpy as np
from tqdm import tqdm

right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
left_chain = [1, 4, 7, 10, 13, 16, 18, 20]


def mirror_joints(ori_path: str, mirror_path: str):
    data = np.load(ori_path)
    assert data.shape[1] == 22 and data.shape[-1] == 3
    data[..., 0] *= -1
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    np.save(mirror_path, data)


if __name__ == '__main__':
    joints_root = 'data/beat/joints_22'
    music_root = 'data/beat/raw_voice'
    for file_name in tqdm(os.listdir(joints_root)):
        mirror_name = 'M' + file_name
        mirror_path = os.path.join(joints_root, mirror_name)
        mirror_joints(os.path.join(joints_root, file_name), mirror_path)
        shutil.copy(os.path.join(music_root, file_name.replace('.npy', '.wav')), os.path.join(music_root, mirror_name.replace('.npy', '.wav')))
