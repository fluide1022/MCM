import glob
import os

import numpy as np
from tqdm import tqdm
import librosa as lr
import soundfile as sf


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    if len(audio) <= window:
        sf.write(f'{out_dir}/{file_name}_slice0.wav', audio, sr)
        return 1, int(audio.shape[0] / sr * 60 + 0.5)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx: start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx, int(audio.shape[0] / sr * 60 + 0.5)


def slice_aistpp(joints_dir, wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    joints = sorted(glob.glob(f"{joints_dir}/*.npy"))
    wav_out = wav_dir + "_sliced"
    joints_out = joints_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(joints_out, exist_ok=True)
    assert len(wavs) == len(joints)
    for wav, joint in tqdm(zip(wavs, joints)):
        # make sure name is matching
        j_name = os.path.splitext(os.path.basename(joint))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert j_name == w_name, str((j_name, wav))
        audio_slices, audio_shape = slice_audio(wav, stride, length, wav_out)
        joints_slices, joints_shape = slice_motion(joint, stride, length, audio_slices, joints_out)
        assert abs(joints_shape - audio_shape) < 2, f'{audio_shape},{joints_shape}'
        # make sure the slices line up
        assert audio_slices == joints_slices, str(
            (wav, joint,  audio_slices, joints_slices, len(wav), len(joint))
        )


def slice_motion(motion_file, stride, length, num_slices, out_dir):
    motion = np.load(motion_file)[:, :22]   # get the former 22 joints

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    num_frames = len(motion)
    # normalize root position
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    if num_frames <= window:
        np.save(f"{out_dir}/{file_name}_slice0.npy", motion)
        return 1, motion.shape[0]
    while start_idx <= num_frames - window and slice_count < num_slices:
        clip = motion[start_idx:start_idx + window]
        np.save(f"{out_dir}/{file_name}_slice{slice_count}.npy", clip)
        start_idx += stride_step
        slice_count += 1
    return slice_count, motion.shape[0]


if __name__ == '__main__':
    # split into 240frames, sample stride is 30 frames
    slice_aistpp('data/aist_plusplus_final/joints_24_fps60',
                 'data/aist_plusplus_final/raw_music', stride=1.0, length=10)
