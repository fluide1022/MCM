import os
from functools import partial
from os.path import join, basename
from pathlib import Path

import librosa
import librosa as lr
import numpy as np
from tqdm import tqdm

FPS = 20
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH
EPS = 1e-6


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


def extract(fpath, skip_completed=True, dest_dir="aist_baseline_feats"):
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = os.path.basename(fpath)
    save_path = os.path.join(dest_dir, audio_name.replace('.wav', '.npy'))

    if os.path.exists(save_path) and skip_completed:
        return

    data, _ = librosa.load(fpath, sr=SR)

    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)

    try:
        start_bpm = _get_tempo(audio_name)
    except:
        # determine manually
        start_bpm = lr.beat.tempo(y=lr.load(fpath)[0])[0]

    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)

    # np.save(save_path, audio_feature)
    return beat_onehot, save_path


def extract_folder(src, dest):
    fpaths = Path(src).glob("*.wav")
    fpaths = sorted(list(fpaths))
    extract_ = partial(extract, skip_completed=False, dest_dir=dest)
    for fpath in tqdm(fpaths):
        rep, path = extract_(fpath)
        jb = np.load(join('data/aist_plusplus_final/jukebox',
                                  basename(path).replace('.wav', '.npy')
                                  ), allow_pickle=True)
        assert abs(rep.shape[0] - jb.shape[0])<2, f'{rep.shape},{jb.shape}'
        rep=rep[:jb.shape[0]]
        np.save(path, rep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--src", default="data/aist_plusplus_final/raw_music",
                        help="source path to AIST++ audio files")
    parser.add_argument("--dest", default="data/aist_plusplus_final/music_beat", help="dest path to audio features")

    args = parser.parse_args()

    extract_folder(args.src, args.dest)
