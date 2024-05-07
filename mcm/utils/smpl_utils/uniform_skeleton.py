import numpy as np
import torch

from utils.smpl_utils.smpl_skeleton import SMPLSkeleton, OFFSETS_DICT, CHAIN_DICT, face_joint_idx, l_idx1, l_idx2


def uniform_skeleton(positions, target_offset, src_skel: SMPLSkeleton = None) -> np.ndarray:
    """
    :param positions: origin positions. t j c
    :param target_offset: j c. offset of reference data
    :param src_skel: skeleton of origin data
    :return:
    """
    num_joints = positions.shape[1]
    if src_skel is None:
        src_skel = SMPLSkeleton(OFFSETS_DICT[num_joints], CHAIN_DICT[num_joints], 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_idx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints
