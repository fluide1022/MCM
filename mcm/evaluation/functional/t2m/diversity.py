from typing import Union

import numpy as np
import torch
from scipy import linalg

def cal_diversity(pred_motion: Union[torch.Tensor, np.ndarray],
                  diversity_times: int):
    """
    :param pred_motion: b t c. All predicted motion vectors from vqvae
    or extracted motion embeddings from eval_wrapper.
    :param diversity_times: randomly choose diversity_times samples from the generated motions.
    :return: calculated diversity
    """
    assert len(pred_motion.shape) == 2
    assert pred_motion.shape[0] > diversity_times
    num_samples = pred_motion.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    diversity = linalg.norm(pred_motion[first_indices] - pred_motion[second_indices], axis=1).mean()
    return diversity
