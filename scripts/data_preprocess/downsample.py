import argparse
import glob
import os

import numpy as np
import scipy.signal
from tqdm import tqdm

"""
    downsample a sequence to target length or fps
"""


def down_sample(x: np.ndarray, tgt_len=None, ori_fps=None, new_fps=None):
    """
    :param x: origin sequence. shape: t ...
    :param tgt_len: target sequence length. If not set, ori_fps and new_fps should not be None
    :param ori_fps: Source fps, if tgt_len set, can be None.
    :param new_fps: Target fps, if tgt_len set, can be None.
    :return: down_sampled sequence
    """
    ori_shape = x.shape

    flatten_x = x.reshape([ori_shape[0], -1])
    # determine the target len

    if tgt_len is None:
        assert ori_fps is not None and new_fps is not None
        tgt_len = int(len(x) * new_fps / ori_fps)

    if ori_shape[0] % tgt_len == 0:
        # simply uniform sample
        y = flatten_x[::(ori_shape[0] // tgt_len)]
    else:
        # resample with scipy
        c = flatten_x.shape[-1]
        y = np.zeros([tgt_len, c])
        for i in range(c):
            y[:, i] = scipy.signal.resample(flatten_x[:, i], tgt_len)

    y = y.reshape([tgt_len, *ori_shape[1:]])

    return y


def down_sample_file(data_path: str, save_path: str, tgt_len=None, ori_fps=None, new_fps=None):
    data: np.ndarray = np.load(data_path)
    data = down_sample(data, tgt_len, ori_fps, new_fps)
    np.save(save_path, data)


if __name__ == '__main__':
    args = argparse.ArgumentParser('down sample a sequence or sequences below a directory')
    args.add_argument('data_root', type=str, help='directory of origin sequences, or a single sequence')
    args.add_argument('save_root', type=str, help='directory to save sequences')
    args.add_argument('--ori_fps', type=int, default=60)
    args.add_argument('--tgt_fps', type=int, default=20)
    args = args.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    if os.path.isfile(args.data_root):
        data_path_list = [args.data_root]
    else:
        data_path_list = glob.glob(os.path.join(args.data_root, '*.npy'))

    for data_path in tqdm(data_path_list):
        save_path = os.path.join(args.save_root, os.path.basename(data_path))
        down_sample_file(data_path, save_path, ori_fps=args.ori_fps, new_fps=args.tgt_fps)
