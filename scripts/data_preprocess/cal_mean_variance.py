import argparse
import sys

import numpy as np
import os
from os.path import join as pjoin

from tqdm import tqdm
sys.path.append(os.curdir)


FEAT_BIAS = 25.0


def mean_variance(vec_dir, save_dir, joints_num):
    file_list = os.listdir(vec_dir)
    data_list = []
    position_begin = 4
    position_end = 4 + (joints_num - 1) * 3
    cont6d_begin = 4 + (joints_num - 1) * 3
    cont6d_end = 4 + (joints_num - 1) * 9
    vel_begin = 4 + (joints_num - 1) * 9
    vel_end = 4 + (joints_num - 1) * 9 + joints_num * 3
    for file in tqdm(file_list):
        index = file.split('.')[0]
        data = np.load(pjoin(vec_dir, file))
        if np.isnan(data).any():
            print(index)
            continue

        data_list.append(data)

    data = np.concatenate(data_list, axis=0)

    Mean = data.mean(axis=0)
    Std = data.std(axis=0)

    def std_average(Std):
        Std[0:1] = Std[0:1].mean() / FEAT_BIAS
        Std[1:3] = Std[1:3].mean() / FEAT_BIAS
        Std[3:4] = Std[3:4].mean() / FEAT_BIAS

        Std[position_begin: position_end] = Std[position_begin: position_end].mean() / 1.0
        Std[cont6d_begin:cont6d_end] = Std[cont6d_begin:cont6d_end].mean() / 1.0
        Std[vel_begin:vel_end] = Std[vel_begin:vel_end].mean() / 1.0
        # feet
        Std[-4:] = Std[-4:].mean() / FEAT_BIAS

        assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]
        return Std

    Std = std_average(Std)
    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std


if __name__ == '__main__':
    args = argparse.ArgumentParser('cal mean and variance')
    args.add_argument('data_dir', type=str, help='root of your data')
    args = args.parse_args()
    vec_dir = os.path.join(args.data_dir, 'vecs_joints_22')
    mean, std = mean_variance(vec_dir, args.data_dir, 22)
