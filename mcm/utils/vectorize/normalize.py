from typing import Union, List

import numpy as np
import torch
from einops import rearrange, repeat


def normalize(motion: Union[torch.Tensor, np.ndarray],
              mean: Union[str, np.ndarray, torch.Tensor],
              std: Union[str, np.ndarray, torch.Tensor],
              return_numpy=False) -> torch.Tensor:
    if isinstance(mean, str):
        mean = np.load(mean)
    if isinstance(std, str):
        std = np.load(std)

    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).to(motion.device)
        std = torch.from_numpy(std).to(motion.device)

    unsqueeze_dim = 0 if len(mean.shape) == 1 else 1
    while len(mean.shape) < len(motion.shape):
        mean = mean.unsqueeze(unsqueeze_dim)
        std = std.unsqueeze(unsqueeze_dim)
    mean = mean.to(motion.device)
    std = std.to(motion.device)
    motion = (motion - mean) / std
    if return_numpy:
        motion = motion.detach().cpu().numpy()
    return motion


def unormalize(motion: Union[torch.Tensor, np.ndarray],
               mean: Union[str, np.ndarray, torch.Tensor],
               std: Union[str, np.ndarray, torch.Tensor],
               return_numpy=False) -> torch.Tensor:
    """
    :param motion: b,t,c or t,c
    :param mean: c or b,c
    :param std: c or b,c
    :return:
    """
    if isinstance(mean, str):
        mean = np.load(mean)
    if isinstance(std, str):
        std = np.load(std)
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean)
        std = torch.from_numpy(std)
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)
    mean = mean.to(motion.device)
    std = std.to(motion.device)
    unsqueeze_dim = 0 if len(mean.shape) == 1 else 1

    while len(mean.shape) < len(motion.shape):
        mean = mean.unsqueeze(unsqueeze_dim)
        std = std.unsqueeze(unsqueeze_dim)
    motion = motion * std + mean
    if return_numpy:
        motion = motion.detach().cpu().numpy()
    return motion


def unormalize_mmseq(mmseq: List[Union[torch.Tensor, str]],
                     modal_list: List[str],
                     mean: Union[str, np.ndarray, torch.Tensor],
                     std: Union[str, np.ndarray, torch.Tensor]):
    for idx, (modal, sub_seq) in enumerate(zip(modal_list, mmseq)):
        if modal == 'motion':
            mmseq[idx] = unormalize(sub_seq, mean, std)
    return mmseq
