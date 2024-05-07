import argparse
import os
import sys
from os.path import join
from typing import List, Union
import torch
import soundfile
import ffmpeg
from mmengine import MODELS, Config
from tqdm import tqdm

sys.path.append(os.curdir)
from utils.vectorize.normalize import unormalize, unormalize_mmseq
from tools.visualize.vis_vec import plot_3d_motion
from utils.files_io.pickle import load_pickle
from utils.smpl_utils.smpl_skeleton import smplh_chain
from utils.vectorize.vec2xyz import recover_from_ric


def vis_seq(save_path: str, seq: List[Union[str, torch.Tensor]], modal_list: List[str], sr=44100):
    """
    visualize a multimodal sequence
    :param modal_list: modal of each sub_sequence in seq
    :param seq: multimodal sequence
    :return: None
    """
    os.makedirs(save_path, exist_ok=True)
    sound_path = join(save_path, 'sound.wav')
    video_path = join(save_path, 'video.mp4')
    video_sound_path = join(save_path, 'video_sound.mp4')
    caption = ""
    motion = torch.empty([0, 623])
    sound = torch.empty([0])
    for sub_seq, modal in zip(seq, modal_list):
        if modal == 'text':
            caption = caption + sub_seq
        elif modal == 'motion':
            sub_seq = sub_seq.detach().cpu().squeeze().to(motion.device)
            caption += '<<Motion_tokens>>'
            motion = torch.cat([motion, sub_seq], dim=0)
        elif modal == 'sound':
            sub_seq = sub_seq.squeeze().to(sound.device)
            caption += '<<Sound_tokens>>'
            sound = torch.cat([sound, sub_seq], dim=0)
        else:
            raise NotImplementedError(f'modal {modal} is not supported')
    motion_exist = motion.shape[0] > 0
    if motion_exist:
        motion = recover_from_ric(motion.to(torch.float32).detach(), 52).numpy()
        plot_3d_motion(video_path, smplh_chain, motion, caption)
    if sound.shape[0] != 0:
        soundfile.write(sound_path, sound, sr)
        if motion_exist:
            input_video = ffmpeg.input(video_path)
            input_audio = ffmpeg.input(sound_path)
            ffmpeg.output(input_video, input_audio, video_sound_path, vcodec='copy', acodec='aac',
                          strict='experimental').run()


if __name__ == '__main__':
    from utils.import_all import *

    args = argparse.ArgumentParser('choose a token pickle to visualise')
    args.add_argument('pkl_path', type=str, help='input a pkl file')
    args.add_argument('--save_root', type=str, default='tmp', help='where to save results')
    args.add_argument('--cfg', type=str, default='configs/llama2_7b_completion.py')
    args = args.parse_args()
    cfg = Config.fromfile(args.cfg)
    tokenizer = MODELS.build(cfg['tokenizer']).cuda()
    data_dict = load_pickle(args.pkl_path)
    basename = os.path.basename(args.pkl_path).split('.')[0]
    mean = data_dict['mean']
    std = data_dict['std']
    for task, prompt_list in tqdm(data_dict.items()):
        for idx, prompt_dict in enumerate(prompt_list):
            # modal_list = prompt_dict['modal_list']
            # seq = prompt_dict['prompt_seq']
            if isinstance(prompt_dict, str):
                continue
            input_ids = prompt_dict['input_ids']
            labels = prompt_dict['labels']
            _, seq, modal_list = tokenizer.decode(input_ids)
            seq = unormalize_mmseq(seq, modal_list, mean, std)
            save_path = os.path.join(args.save_root, basename, f'{task}_{idx}')
            vis_seq(save_path, seq, modal_list)
