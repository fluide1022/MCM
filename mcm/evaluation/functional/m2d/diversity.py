import numpy as np
from scipy import linalg

def cal_diversity(pred_motion):
    bs = pred_motion.shape[0]
    dist = 0
    for i in range(bs):
        for j in range(i + 1, bs):
            dist += np.linalg.norm(pred_motion[i] - pred_motion[j])
    dist /= bs * (bs - 1) / 2
    return dist
