import random
from collections import defaultdict
from os.path import join, exists
from typing import Dict

import numpy as np
import torch
from mmengine import Config, DATASETS
from torch.utils.data import Dataset
from tqdm import tqdm

from mcm.evaluation.functional.t2m.word_vectorizer import WordVectorizer
from mcm.utils.files_io.txt import read_list_txt
from mcm.utils.vectorize.normalize import normalize


def group_and_select_random(tuples_list):
    grouped_dict = defaultdict(list)

    for tuple_item in tuples_list:
        grouped_dict[tuple_item[1]].append(tuple_item)

    selected_elements = [random.choice(group) for group in grouped_dict.values()]

    return selected_elements

@DATASETS.register_module(force=True)
class EvalT2MDataset(Dataset):
    def __init__(self,
                 data_root: str = 'data/humanml3d',
                 index_file: str = 'test.txt',
                 caption_prefix: str = 'texts',
                 motion_prefix: str = 'vecs_joints_22',
                 eval_cfg: Config = None,
                 mean_file: str = 'data/humanml3d/Mean.npy',
                 std_file: str = 'data/humanml3d/Std.npy'):
        # only select one caption for one motion
        self.data_root = data_root
        self.index_file = join(data_root, index_file)
        self.motion_prefix = motion_prefix
        self.caption_prefix = caption_prefix
        self.cfg = eval_cfg
        self.mean_file = mean_file
        self.std_file = std_file

        self.w_vectorizer = WordVectorizer(eval_cfg['glove_path'], 'our_vab')
        self.max_length = 20
        self.pointer = 0
        self.consider_range = (40, 200)  # close range <40 >199 will be ignored
        self.get_data_info()
        length_list = [self.data_dict[name]['m_length'] for name in self.name_list]
        # sort name by motion length
        self.name_list, length_list = zip(*sorted(zip(self.name_list, length_list), key=lambda x: x[1]))
        self.length_array = np.asarray(length_list)
        self.reset_max_len(self.max_length)

        print(str(self))

    def __str__(self):
        return f'{len(self)} samples in for the test dataset'
    def get_data_info(self):
        self.data_dict=dict()
        self.name_list = []
        id_list = []
        for line in read_list_txt(self.index_file):
            id_list.append(line.strip())
        for name in tqdm(id_list):
            motion_path = join(self.data_root, self.motion_prefix, name + '.npy')
            if not exists(motion_path):
                continue
            motion = np.load(motion_path)
            # [40,199] conventionally
            if (len(motion)) < self.consider_range[0] or (len(motion) > self.consider_range[1]):
                continue
            texts, start_end_frames = self.fetch_text(name, motion)
            if texts is not None and len(texts) != 0:
                def choose_texts():
                    text_list = [(text, start_end) for text, start_end in zip(texts, start_end_frames)]
                    text_list = group_and_select_random(text_list)
                    return [item[0] for item in text_list], [item[1] for item in text_list]

                texts, start_end_frames = choose_texts()
            if texts is not None and len(texts) != 0:
                # multi captions for one motion
                for t_idx, t in enumerate(texts):
                    new_name = f'{name}_{t_idx}'
                    cropped_motion = self.crop_motion(motion, start_end_frames[t_idx])
                    if len(cropped_motion) > self.consider_range[1] or len(cropped_motion) < self.consider_range[0]:
                        continue
                    self.data_dict[new_name] = {
                        'motion': cropped_motion,  # content
                        'm_length': len(cropped_motion),
                        'caption': t,  # path
                    }
                    self.name_list.append(new_name)

    def reset_max_len(self, length):
        # don't know why. maybe filter motions below 20?
        assert length <= self.cfg.max_motion_len
        # search 20 in all motion lengths
        self.pointer = np.searchsorted(self.length_array, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def fetch_text(self, name: str, motion: np.ndarray):
        text_data = []
        start_end_frames = []
        text_path = join(self.data_root, self.caption_prefix, name + '.txt')
        if not exists(text_path):
            return None, None
        for line in read_list_txt(text_path):
            text_dict = {}
            line_split = line.strip().split('#')
            text = line_split[0]
            tokens = line_split[1].split(' ')
            f_tag = float(line_split[2])
            to_tag = float(line_split[3])
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            text_dict['text'] = text
            text_dict['tokens'] = tokens
            if f_tag == 0.0 and to_tag == 0.0:
                start_end_frames.append((0, len(motion)))
            else:
                start_end_frames.append((int(f_tag * 20), int(to_tag * 20)))
            text_data.append(text_dict)
        return text_data, start_end_frames
    def crop_motion(self, motion: np.ndarray, start_end_frame=(0, None)):
        start, end = start_end_frame
        if end is None:
            end = len(motion)
        return motion[start:end]
    def fetch_text_info(self, text_data: Dict = None, m_length=None):
        if text_data is None:
            return None, None, None, None, m_length
        text = text_data['text']

        tokens = text_data['tokens']
        if len(tokens) < self.cfg.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.cfg.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.cfg.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        # Crop the motions in to times of 4, and introduce small variations

        if self.cfg.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        if coin2 == 'double':
            m_length = (m_length // self.cfg.unit_length - 1) * self.cfg.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.cfg.unit_length) * self.cfg.unit_length
        return text, torch.from_numpy(word_embeddings), torch.from_numpy(pos_one_hots), sent_len, m_length
    def pad_or_crop_array(self, tgt_len, seq: np.ndarray = None):
        """
        :param tgt_len: int
        :param seq: np.ndarray [t,...]
        :return: if t<tgt_len, add padding zeros, otherwise, crop to tgt_len
        """
        if seq is None:
            return None
        if len(seq) >= tgt_len:
            return seq[:tgt_len]
        padded_seq = np.zeros([tgt_len, *seq.shape[1:]])
        padded_seq[:len(seq)] = seq
        return padded_seq

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_data = data['motion'], data['m_length'], data['caption']
        max_motion_len = self.cfg.max_motion_len
        # text, m_len modification for evaluation
        text, word_embeddings, pos_one_hots, sent_len, m_length = self.fetch_text_info(text_data, m_length)

        "Z Normalization"
        motion = normalize(motion, self.mean_file, self.std_file)

        # crop. as in motion_diffuse and mdm, remain the original m_length after crop,
        # maybe it's useful during evaluation
        if m_length > self.cfg.max_motion_len:
            start_frame = random.randint(0, len(motion) - max_motion_len)
            motion = motion[start_frame: start_frame + max_motion_len]
        else:
            motion = self.pad_or_crop_array(max_motion_len, motion)

        assert len(motion) == max_motion_len, f'{len(motion)},{max_motion_len}'
        # for beat alignment calculation
        item_dict={
            'word_embeddings': word_embeddings,
            'pos_one_hots': pos_one_hots,
            'sent_len': sent_len,
            'motion': torch.Tensor(motion),
            'm_length': m_length,
            'caption': text,
        }
        for key, value in item_dict.items():
            assert not isinstance(value, np.ndarray), key
        return item_dict


