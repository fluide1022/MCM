from typing import List, Union

import librosa
import numpy as np
import torch
from librosa.onset import onset_strength
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

def _get_tempo(audio_name):
    """Get tempo (BPM) for a music by parsing music name."""

    # a lot of stuff, only take the 5th element
    audio_name = audio_name.split("_")[4]

    assert len(audio_name) == 4
    if audio_name[0:3] in [
        "mBR",
        "mPO",
        "mLO",
        "mMH",
        "mLH",
        "mWA",
        "mKR",
        "mJS",
        "mJB",
    ]:
        return int(audio_name[3]) * 10 + 80
    elif audio_name[0:3] == "mHO":
        return int(audio_name[3]) * 5 + 110
    elif audio_name[1:4] in [
        "MmBR",
        "MmPO",
        "MmLO",
        "MmMH",
        "MmLH",
        "MmWA",
        "MmKR",
        "MmJS",
        "MmJB",
    ]:
        return int(audio_name[4]) * 10 + 80
    elif audio_name[1:4] == "mHO":
        return int(audio_name[4]) * 5 + 110
    else:
        assert False, audio_name

def cal_beat(raw_music:np.ndarray, fps=20, sr=44100, audio_name=None):
    """
    :param raw_music: t(sr * t_sec)
    :return: [beat list]
    """

    # 提取节拍
    if audio_name is None:
        tempo, beats = librosa.beat.beat_track(y=raw_music, sr=sr)

        #
        beat_times = librosa.frames_to_time(beats, sr=sr)
        beat_times = beat_times * fps
    else:
        envelope = librosa.onset.onset_strength(y=raw_music, sr=sr)
        try:
            start_bpm = _get_tempo(audio_name)
        except:
            start_bpm = librosa.beat.tempo(y=raw_music)[0]
        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope,
            sr=sr,
            start_bpm=start_bpm,
            tightness=100,
        )
        beat_idxs = librosa.frames_to_time(beat_idxs, sr=sr) * fps
        return beat_idxs

    return beat_times

def cal_beat_align(pred_motion:Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor,np.ndarray]]],
                   music_beat_batch:List[np.ndarray],
                   m_len:Union[np.ndarray, List[int], ]):
    """
    :param pred_motion: bs t j c
    :param music_beat_feature: bs * np.ndarray[num_beat]
    :param m_len: bs
    :return:
    """

    dance_beat_batch = cal_dance_beat_batch(pred_motion, m_len)
    ba_score = []
    for dance_beat, music_beat in zip(dance_beat_batch, music_beat_batch):
        ba = 0.
        for mb in music_beat:
            ba += np.exp(-np.min((dance_beat - mb) ** 2) / 18)
        ba_score.append(ba / len(music_beat))
    return np.mean(ba_score)


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
    # upsample to 60fps to make a fair comparison
    pred_motion = upsample_motion(pred_motion, 3)
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

def upsample_motion(data, factor=3):
    """
    使用双线性插值上采样时间序列数据。

    :param data: 形状为 (t, j, 3) 的时间序列数据。
    :param factor: 上采样因子，默认为 3。
    :return: 上采样后的时间序列数据。
    """
    t, j, _ = data.shape
    # 创建新的时间点
    new_time_points = np.linspace(0, t-1, t*factor)

    # 初始化上采样后的数据数组
    upsampled_data = np.zeros((t*factor, j, 3))

    # 对每个关节和每个维度进行插值
    for joint in range(j):
        for dim in range(3):
            interp_func = interp1d(np.arange(t), data[:, joint, dim], kind='linear')
            upsampled_data[:, joint, dim] = interp_func(new_time_points)

    return upsampled_data

if __name__ == '__main__':
    pred_motion = torch.rand([32, 60, 22, 3])
    music_beat_feature = torch.randint(size=[32, 60], low=0, high=2)
    print(cal_beat_align(pred_motion, music_beat_feature))
