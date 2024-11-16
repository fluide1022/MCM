import os.path
import random
from os.path import join
from typing import List, Union, Tuple

import numpy as np
import torch
from mmengine import Config
from torch.utils.data import Dataset
import sys

from tqdm import tqdm

from mcm.datasets.build_dataloader import build_dataloader
from mcm.registry import DATASETS
from mcm.utils.files_io.txt import read_list_txt
from mcm.utils.vectorize.normalize import normalize
from scripts.data_preprocess.humanml3d.caption_process import analyze_caption

sys.path.append(os.curdir)

@DATASETS.register_module(force=True)
class MotionTextDataset(Dataset):
    def __init__(self,
                 data_root: Union[str, List[str]],
                 motion_prefix: Union[str, List[str]],
                 caption_prefix: Union[str, List[str]],
                 index_file: Union[str, List[str]],
                 mean_file: Union[str, List[str]],
                 std_file: Union[str, List[str]],
                 ignore_file: Union[str, List[str]],
                 max_motion_length: int = 196,
                 repeat: Union[int, List[int]] = 1
                 ):
        """ A dataset for loading motion vectors for a single or multiple datasets
        :param data_root: exp: 'data/humanml3d' or ['data/humanml3d','data/aist']
        :param motion_prefix: the folder where save motion vectors for each data_root
        :param index_file: exp: 'train.txt' or ['train.txt', 'train_val.txt'], responding to data_root
        :param mean_file: Mean.npy file for each dataset or a union file for all datasets.
        :param std_file: Std.npy file for each dataset or a union file for all datasets
        :param window_size: clip each total sequence into window size clips. in frame.
        :param have_hand: json file which stored if the hands of motion is synthesised.
                            True means the hands are real, False means synthesised, then
                            hands will be ignored when calculating reconstruction loss.
        :param ignore_file: txt file stored indexes needs to be ignored.
        60 frames of 20 fps by default
        """
        self.data_root = data_root
        self.motion_prefix = motion_prefix
        self.index_file = index_file
        self.mean_file = mean_file
        self.std_file = std_file
        self.ignore_file = ignore_file
        self.max_motion_length = max_motion_length
        self.repeat = repeat
        self.caption_prefix = caption_prefix

        self.map_dataset_info()
        self.get_data_info()
        self.__str__()

    def map_dataset_info(self):
        if isinstance(self.data_root, str):
            self.data_root = [self.data_root]
        num_datasets = len(self.data_root)

        if isinstance(self.motion_prefix, str):
            # relative to absolute
            self.motion_prefix = [self.motion_prefix] * num_datasets

        if isinstance(self.caption_prefix, str):
            self.caption_prefix = [self.caption_prefix] * num_datasets

        if isinstance(self.index_file, str):
            self.index_file = [self.index_file] * num_datasets

        if isinstance(self.mean_file, str):
            self.mean_file = [self.mean_file] * num_datasets

        if isinstance(self.std_file, str):
            self.std_file = [self.std_file] * num_datasets

        if isinstance(self.ignore_file, str):
            self.ignore_file = [self.ignore_file] * num_datasets

        if hasattr(self, 'repeat') and isinstance(self.repeat, int):
            self.repeat = [self.repeat] * num_datasets

        self.motion_prefix = [os.path.join(self.data_root[i], self.motion_prefix[i])
                              for i in range(num_datasets)]
        self.caption_prefix = [os.path.join(self.data_root[i], self.caption_prefix[i])
                               for i in range(num_datasets)]
        self.index_file = [os.path.join(self.data_root[i], self.index_file[i])
                           for i in range(num_datasets)]

        self.ignore_file = [os.path.join(self.data_root[i], self.ignore_file[i])
                            for i in range(num_datasets)]

        for i in range(num_datasets):
            # judge if mean_file and std_file should be joined with data_root
            if not os.path.exists(self.mean_file[i]):
                self.mean_file[i] = os.path.join(self.data_root[i], self.mean_file[i])
                self.std_file[i] = os.path.join(self.data_root[i], self.std_file[i])

    def get_data_info(self):
        """
        :return: a dict for unified data loading
        """
        self.data_dict = dict()
        for data_root, motion_dir, caption_root, idx_file, mean_npy, std_npy, ignore_list, repeat_times in \
                zip(self.data_root, self.motion_prefix, self.caption_prefix, self.index_file,
                    self.mean_file, self.std_file, self.ignore_file, self.repeat):
            dataset_name = os.path.basename(os.path.normpath(data_root))
            if not os.path.exists(ignore_list):
                ignore_list = []
            else:
                ignore_list = read_list_txt(ignore_list)
            for idx in tqdm(read_list_txt(idx_file), desc=dataset_name):

                if idx in ignore_list:
                    continue
                data_path = os.path.join(motion_dir, idx + '.npy')
                if not os.path.exists(data_path):
                    print(data_path, ' not exist')
                    continue

                motion = np.load(data_path)

                lines = read_list_txt(join(caption_root, idx + '.txt'))
                # get text
                for line_idx, line in enumerate(lines):

                    raw_caption, (f_tag, to_tag), _ = analyze_caption(line)
                    if to_tag == 0:
                        to_tag = len(motion)

                    motion_len = to_tag - f_tag

                    if dataset_name == 'humanml3d' and (motion_len > 200 or motion_len < 24):
                        print(f'{idx}_{line_idx}')
                        continue

                    self.data_dict[f'{idx}_{line_idx}'] = {'motion_path': data_path,
                                                           'm_length': to_tag - f_tag,
                                                           'text': raw_caption,
                                                           'from': f_tag,
                                                           'to': to_tag,
                                                           'mean': mean_npy,
                                                           'std': std_npy}

        self.data_list = list(self.data_dict.keys())

    def load_item(self, item):
        """
        :param offset: return motion[offset:offset+self.window_size].
                    if None, offset will be randomly set.
        :param item: index in self.data_list
        :return: motion in npy. shape in [t c]
        """
        key = self.data_list[item]
        info = self.data_dict[key]
        mean = info['mean']
        std = info['std']
        motion = np.load(info['motion_path'])[info['from']:info['to']]
        # as same as in motion diffuse. m_length is length of motion before crop
        m_length = len(motion)
        if len(motion) > self.max_motion_length:
            info['from'] = random.randint(0, len(motion) - self.max_motion_length)
            info['to'] = info['from'] + self.max_motion_length
            motion = motion[info['from']: info['to']]
        motion = normalize(motion, mean, std)
        assert motion is not None
        info.update(
            motion=motion.to(torch.float32),
            mean=mean,
            std=std,
            m_length=m_length
        )
        return info

    def __getitem__(self, item):
        data_dict = self.load_item(item)
        return data_dict

    def __len__(self):
        return len(self.data_list)

    def __str__(self):
        print('Dataset for motion-text training!')
        print(f'Dataset contains {len(self.data_root)}')
        print(f'path of each data root is:')
        print(f'{self.data_root}')
        print(f'There are {len(self)} motion samples in total.')


if __name__ == '__main__':
    cfg = Config.fromfile('configs/mcm/mcm_no_control.py')['train_dataloader']
    data_loader = build_dataloader(cfg)
    for batch in tqdm(data_loader):
        print(batch.keys())
