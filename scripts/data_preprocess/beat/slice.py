import glob
import os
import sys

import librosa
import numpy as np
from tqdm import tqdm
import librosa as lr
import soundfile as sf

sys.path.append(os.curdir)

SR = 16000


def load_get_length(audio_file, motion_file):
    audio, sr = lr.load(audio_file, sr=None)
    motion = np.load(motion_file)
    return audio, motion, sr


def slice_audio(audio: np.ndarray, stride, length, sr):
    """
    :param audio: audio array shape in (time*sr,)
    :param stride: unit is sec
    :param length: unit is sec
    :param time_motion: num of seconds of the motion.
    :return:
    """
    # stride, length in seconds
    start_idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    if len(audio) <= window:
        return [audio]
    audio_slices = []
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx: start_idx + window]
        audio_slices.append(audio_slice)
        start_idx += stride_step
    return audio_slices


def slice_beat(joints_dir, wav_dir, stride=60, length=60):
    joints = sorted(glob.glob(f"{joints_dir}/*.npy"))
    wav_out = wav_dir + "_sliced"
    joints_out = joints_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(joints_out, exist_ok=True)
    for joint_path in tqdm(joints):
        file_name = os.path.splitext(os.path.basename(joint_path))[0]
        # make sure name is matching
        wav_path = os.path.join(wav_dir, os.path.basename(joint_path).replace('.npy', '.wav'))
        motion_arr = np.load(joint_path)
        audio_arr, _ = librosa.load(wav_path, sr=SR)
        sec = min(motion_arr.shape[0] / 20, audio_arr.shape[0] / SR)

        motion_arr = motion_arr[:int(20 * sec)]
        audio_arr = audio_arr[:int(SR * sec)]

        joints_slices = slice_motion(motion_arr, stride, length)
        audio_slices = slice_audio(audio_arr, stride, length, sr=SR)

        assert len(audio_slices) == len(joints_slices), \
            f'{wav_path},{len(audio_slices)} , {len(joints_slices)}'
        # means no words in the last few seconds
        for slice_id, (sa, sj) in enumerate(zip(audio_slices, joints_slices)):
            np.save(f"{joints_out}/{file_name}_slice{slice_id}.npy", sj)
            sf.write(f"{wav_out}/{file_name}_slice{slice_id}.wav", sa, samplerate=SR)


def slice_motion(motion, stride, length, fps=20):
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
    slice_beat('data/beat/joints_22',
               'data/beat/raw_voice',
               stride=60, length=60)
