import glob
import os
import random
import sys
import numpy as np
from textgrid import TextGrid
from tqdm import tqdm
import librosa as lr
import soundfile as sf
import textgrid

from scripts.data_preprocess.aist.gen_text import get_token

sys.path.append(os.curdir)

TEXT_PREFIX = [
    'a {} speaker is saying the following content, "{}"',
    'a {} uttered the following words, "{}"',
    'a {} said, "{}"'
]


def get_gender(filename: str):
    MALE_LIST = list(range(1, 6)) + list(range(11, 21))
    speaker_id = int(filename.split('_')[0])
    if speaker_id in MALE_LIST:
        return 'male'
    return 'female'


def load_get_length(audio_file, tg_file, motion_file, motion_fps):
    audio, sr = lr.load(audio_file, sr=None)
    audio_time = len(audio) / sr
    tg = textgrid.TextGrid()
    tg.read(tg_file)
    tg_time = tg.maxTime
    motion = np.load(motion_file)
    motion_time = len(motion) / motion_fps
    return audio, tg, motion, sr, min([audio_time, tg_time, motion_time])


def slice_audio(audio: np.ndarray, stride, length, sr, target_time: float):
    """
    :param audio: audio array shape in (time*sr,)
    :param stride: unit is sec
    :param length: unit is sec
    :param time_motion: num of seconds of the motion.
    :return:
    """
    # stride, length in seconds
    audio = audio[:int(sr * target_time)]
    start_idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    if len(audio) <= window:
        return [audio], sr
    audio_slices = []
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx: start_idx + window]
        audio_slices.append(audio_slice)
        start_idx += stride_step
    return audio_slices


def slice_text_grid(tg: TextGrid, stride, length, target_time):
    audio_len = tg.maxTime
    audio_len = min(audio_len, target_time)
    words = tg.tiers[0]
    cur_start = 0.
    cur_slice_words = []
    sentences = []
    while cur_start < audio_len:
        cur_end = cur_start + length
        if cur_end > audio_len:
            break
        for word_tuple in words:
            start, word = word_tuple.minTime, word_tuple.mark
            if start >= cur_end:
                break
            elif start < cur_start:
                continue
            if len(word):
                cur_slice_words.append(word)
        sentences.append(cur_slice_words)
        cur_slice_words = []
        cur_start += stride
    return sentences


def slice_beat(joints_dir, text_grid_dir, wav_dir, stride=5., length=10):
    joints = sorted(glob.glob(f"{joints_dir}/*.npy"))
    wav_out = wav_dir + "_sliced"
    joints_out = joints_dir + "_sliced"
    text_out = os.path.join(os.path.dirname(os.path.normpath(text_grid_dir)), 'texts')
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(text_out, exist_ok=True)
    os.makedirs(joints_out, exist_ok=True)
    for joint_path in tqdm(joints):
        file_name = os.path.splitext(os.path.basename(joint_path))[0]
        gender = get_gender(file_name)
        # make sure name is matching
        wav_path = os.path.join(wav_dir, os.path.basename(joint_path).replace('.npy', '.wav'))
        tg_path = os.path.join(text_grid_dir, os.path.basename(joint_path).replace('.npy', '.TextGrid'))
        audio_arr, tg, motion_arr, sr, target_time = load_get_length(wav_path, tg_path, joint_path, 120)
        joints_slices = slice_motion(motion_arr, stride, length, target_time, 120)
        audio_slices = slice_audio(audio_arr, stride, length, sr, target_time)
        text_slices = slice_text_grid(tg, stride, length, target_time)

        assert len(audio_slices) == len(joints_slices) and len(audio_slices) - len(text_slices) <= 1, \
            f'{wav_path},{len(audio_slices)} , {len(text_slices)}, {len(joints_slices)}'
        # means no words in the last few seconds
        if len(audio_slices) - len(text_slices) == 1:
            audio_slices = audio_slices[:-1]
            joints_slices = joints_slices[:-1]
        for slice_id, (sa, st, sj) in enumerate(zip(audio_slices, text_slices, joints_slices)):
            if len(st):
                st = ' '.join(st)
                st = random.choice(TEXT_PREFIX).format(gender, st)
                token = get_token(st)
                row = st + '.#' + token + '#0.0#0.0\n'
                with open(f"{text_out}/{file_name}_slice{slice_id}.txt", 'w') as fp:
                    fp.write(row)
                np.save(f"{joints_out}/{file_name}_slice{slice_id}.npy", sj)
                sf.write(f"{wav_out}/{file_name}_slice{slice_id}.wav", sa, sr)


def slice_motion(motion, stride, length, target_time, fps=120):
    motion = motion[:int(target_time * fps)]
    num_frames = len(motion)
    # normalize root position
    start_idx = 0
    window = int(length * fps)
    stride_step = int(stride * fps)
    slice_count = 0
    # slice until done or until matching audio slices
    if num_frames <= window:
        return [motion]
    motions = []
    while start_idx <= num_frames - window:
        clip = motion[start_idx:start_idx + window]
        motions.append(clip)
        start_idx += stride_step
        slice_count += 1
    return motions


if __name__ == '__main__':
    # split into 240 frames, sample stride is 30 frames
    slice_beat('data/beat/smpl',
               'data/beat/raw_text_grid',
               'data/beat/raw_voice',
               stride=5, length=10)
