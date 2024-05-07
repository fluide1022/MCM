import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa.display import waveshow, specshow

args = argparse.ArgumentParser()
args.add_argument('--audio_file', default='data/aist_plusplus_final/raw_music/gBR_sBM_cAll_d04_mBR0_ch04.wav')
args.add_argument('--wave', action='store_true')
args = args.parse_args()
# 读取音频文件
audio_file = args.audio_file

y, sr = librosa.load(audio_file)

# 计算短时傅里叶变换（STFT）
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

# 绘制彩色的波形图
plt.figure(figsize=(10, 6))
if args.wave:
    waveshow(D, sr=sr)
    plt.title('Wave')
else:
    specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
plt.savefig(audio_file.replace('.wav', '.png'))
