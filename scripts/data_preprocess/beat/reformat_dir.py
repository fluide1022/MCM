import glob
import os
import shutil

from tqdm import tqdm

if __name__ == '__main__':
    bvh_root = 'data/beat/bvh'
    csv_root = 'data/beat/csv'
    emotion_root = 'data/beat/emotion'
    text_root = 'data/beat/raw_text'
    text_grid = 'data/beat/raw_text_grid'
    raw_root = 'data/beat/raw'
    voice_root = 'data/beat/raw_voice'
    os.makedirs(voice_root, exist_ok=True)
    os.makedirs(bvh_root, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)
    os.makedirs(emotion_root, exist_ok=True)
    os.makedirs(text_root, exist_ok=True)
    os.makedirs(text_grid, exist_ok=True)
    for speaker in tqdm(os.listdir(raw_root)):
        for file in glob.glob(os.path.join(raw_root, speaker, '*')):
            if file.endswith('bvh'):
                if not os.path.exists(os.path.join(bvh_root,os.path.basename(file))):
                    shutil.move(file, bvh_root)
            elif file.endswith('csv'):
                if not os.path.exists(os.path.join(csv_root, os.path.basename(file))):
                    shutil.move(file, csv_root)
            elif file.endswith('text'):
                if not os.path.exists(os.path.join(text_root, os.path.basename(file))):
                    shutil.move(file, text_root)
            elif file.endswith('json'):
                if not os.path.exists(os.path.join(emotion_root, os.path.basename(file))):
                    shutil.move(file, emotion_root)
            elif file.endswith('.TextGrid'):
                if not os.path.exists(os.path.join(text_grid, os.path.basename(file))):
                    shutil.move(file, text_grid)
            elif file.endswith('.wav'):
                if not os.path.exists(os.path.join(voice_root, os.path.basename(file))):
                    shutil.move(file, voice_root)
