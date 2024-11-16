import os
from collections import defaultdict

import librosa
import torch
from mmengine import DATASETS, Config
from torch.utils import data
import numpy as np

from os.path import join, basename
import random
from tqdm import tqdm

from mcm.datasets.build_dataloader import build_dataloader
from mcm.evaluation.functional.m2d.beat_alignment import cal_beat
from mcm.utils.files_io.txt import read_list_txt
from mcm.utils.vectorize.normalize import normalize
from scripts.data_preprocess.humanml3d.caption_process import analyze_caption


def group_and_select_random(tuples_list):
    grouped_dict = defaultdict(list)

    for tuple_item in tuples_list:
        grouped_dict[tuple_item[1]].append(tuple_item)

    selected_elements = [random.choice(group) for group in grouped_dict.values()]

    return selected_elements


@DATASETS.register_module(force=True)
class EvalM2DDataset(data.Dataset):
    """Dataset for music-to-dance and co-speech generation task.

    """

    def __init__(self,
                 data_root: str,
                 motion_prefix: str,
                 caption_prefix: str,
                 audio_prefix: str,
                 jukebox_prefix: str,
                 index_file: str,
                 mean_file: str,
                 std_file: str,
                 ignore_file: str,
                 clip_length: int = 80,
                 step: int = 40,
                 sr=44100,
                 **kwargs
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
        """
        self.data_root = data_root
        self.motion_prefix = motion_prefix
        self.index_file = index_file
        self.mean_file = mean_file
        self.std_file = std_file
        self.ignore_file = ignore_file
        self.audio_prefix = audio_prefix
        self.caption_prefix = caption_prefix
        self.jukebox_prefix = jukebox_prefix
        self.sr = sr
        self.map_dataset_info()
        self.get_data_info()
        print(str(self))

    def map_dataset_info(self):

        self.motion_prefix = os.path.join(self.data_root, self.motion_prefix)
        self.audio_prefix = os.path.join(self.data_root, self.audio_prefix)
        self.caption_prefix = os.path.join(self.data_root, self.caption_prefix)
        self.index_file = os.path.join(self.data_root, self.index_file)
        self.jukebox_prefix = os.path.join(self.data_root, self.jukebox_prefix)
        self.ignore_file = os.path.join(self.data_root, self.ignore_file)
        # judge if mean_file and std_file should be joined with data_root
        if not os.path.exists(self.mean_file):
            self.mean_file = os.path.join(self.data_root, self.mean_file)
            self.std_file = os.path.join(self.data_root, self.std_file)

    def get_data_info(self):
        """
        :return: a dict for unified data loading
        """
        self.data_dict = dict()

        dataset_name = os.path.basename(os.path.normpath(self.data_root))
        if not os.path.exists(self.ignore_file):
            ignore_list = []
        else:
            ignore_list = read_list_txt(self.ignore_file)
        for idx in tqdm(read_list_txt(self.index_file), desc=dataset_name):
            if idx in ignore_list:
                continue
            data_path = os.path.join(self.motion_prefix, idx + '.npy')
            if not os.path.exists(data_path):
                print(data_path, ' not exist')
                continue

            motion = np.load(data_path)

            jukebox_path = join(self.jukebox_prefix, idx + '.npy')
            audio_path = join(self.audio_prefix, idx + '.wav')
            line = read_list_txt(join(self.caption_prefix, idx + '.txt'))[0]
            # get text
            raw_caption, _, _ = analyze_caption(line)
            self.data_dict[f'{idx}'] = {
                'motion_path': data_path,
                'jukebox_path': jukebox_path,
                'audio_path': audio_path,
                'text': raw_caption,
                'mean': self.mean_file,
                'std': self.std_file
            }

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
        motion = np.load(info['motion_path'])
        jukebox = np.load(info['jukebox_path'])
        audio = librosa.load(info['audio_path'], sr=self.sr)[0]
        min_len = min(len(motion), len(jukebox))
        assert len(motion) < len(jukebox)
        motion = motion[:min_len]
        jukebox = jukebox[:min_len]
        assert len(motion) == len(jukebox)
        # same to bailando, FACT ... avoiding unfair comparison
        beat = cal_beat(audio, fps=60, sr=self.sr, audio_name=basename(info['audio_path']))
        assert not np.any(np.isnan(motion)), key
        motion = normalize(motion, mean, std)
        assert motion is not None
        info.update(
            beat=beat,
            audio=audio,
            motion=motion,
            jukebox=torch.from_numpy(jukebox).to(torch.float32),
            mean=mean,
            std=std,
            m_length=len(motion)
        )
        return info

    def __getitem__(self, item):
        data_dict = self.load_item(item)
        return data_dict

    def __len__(self):
        return len(self.data_list)

    def __str__(self):
        return 'Dataset for aist++ evaluation!\n' \
               f'{self.data_root}\n' \
               f'There are {len(self)} motion samples in total.'
