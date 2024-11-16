import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema


def cal_beat_align(pred_motion, music_beat_feature, m_len):
    """
    :param pred_motion: bs t j c
    :param music_beat_feature: bs t
    :param m_len: bs
    :return:
    """
    if isinstance(pred_motion, torch.Tensor):
        pred_motion = pred_motion.detach().cpu().numpy()
    if isinstance(music_beat_feature, torch.Tensor):
        music_beat_feature = music_beat_feature.detach().cpu().numpy()
    dance_beat_batch = cal_dance_beat_batch(pred_motion, m_len)
    music_beat_batch = cal_music_beat_batch(music_beat_feature, m_len)
    ba_score = []
    for dance_beat, music_beat in zip(dance_beat_batch, music_beat_batch):
        ba = 0.
        for mb in music_beat:
            ba += np.exp(-np.min((dance_beat - mb) ** 2) / 18)
        ba_score.append(ba / len(music_beat))
    return np.sum(ba_score)


def cal_dance_beat_batch(pred_motion, m_len):
    """
    :param pred_motion: b t j c
    :return: a list of beat frames of every sample in batch
    """
    batch_dance_beat = [cal_dance_beat(m[:l])[0] for m, l in zip(pred_motion, m_len)]
    return batch_dance_beat


def cal_dance_beat(pred_motion):
    """
    :param pred_motion: t j c
    :return:
    """
    kinetic_vel = np.mean(np.sqrt(np.sum((pred_motion[1:] - pred_motion[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = gaussian_filter(kinetic_vel, 5)
    # find the minimum point of velocity return (array,)
    motion_beats = argrelextrema(kinetic_vel, np.less)[0]
    return motion_beats, len(kinetic_vel)


def cal_music_beat_batch(music_beat_feature: np.ndarray, m_len):
    """
    :param music_beat_feature: bs t
    :param m_len: bs
    :return: a list of beat frames of every sample in batch
    """
    batch_music_beat = [cal_music_beat(m[:l]) for m, l in zip(music_beat_feature, m_len)]
    return batch_music_beat


def cal_music_beat(music_beat_feature: np.ndarray):
    """
    :param music_beat_feature: T, 0,1 distribution
    :return: list of beat frames
    """
    music_beat_feature = music_beat_feature.astype(bool)
    beat_axis = np.arange(len(music_beat_feature))
    beat_axis = beat_axis[music_beat_feature]
    return beat_axis


if __name__ == '__main__':
    pred_motion = torch.rand([32, 60, 22, 3])
    music_beat_feature = torch.randint(size=[32, 60], low=0, high=2)
    print(cal_beat_align(pred_motion, music_beat_feature))
