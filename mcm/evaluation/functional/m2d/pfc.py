import numpy as np

DT = 1/20.
up_dir = 1  # y is up
flat_dirs = [i for i in range(3) if i != up_dir]
def cal_pfc(pred_motion_batch):
    scores = np.mean([cal_pfc_single(pred_motion) for pred_motion in pred_motion_batch])
    return scores*10000


def cal_pfc_single(pred_motion):
    """
    :param pred_motion: t,j,c
    :return: float
    """
    root_v = (pred_motion[1:, 0, :] - pred_motion[:-1, 0, :]) / DT  # root velocity (S-1, 3)
    root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
    # clamp the up-direction of root acceleration
    root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
    # l2 norm
    root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
    scaling = root_a.max()
    root_a /= scaling

    foot_idx = [7, 10, 8, 11]
    feet = pred_motion[:, foot_idx]  # foot positions (S, 4, 3)
    foot_v = np.linalg.norm(
        feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
    )  # (S-2, 4) horizontal velocity
    foot_mins = np.zeros((len(foot_v), 2))
    foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
    foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

    foot_loss = (
            foot_mins[:, 0] * foot_mins[:, 1] * root_a
    )  # min leftv * min rightv * root_a (S-2,)
    foot_loss = foot_loss.mean()
    return foot_loss