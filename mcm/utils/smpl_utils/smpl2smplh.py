import argparse
import os.path
import sys
from typing import Union

import numpy as np
import torch
from einops import rearrange
from smplx import SMPLH

sys.path.append(os.curdir)
from utils.smpl_utils.smpl_ik import SMPL_IK

from utils.smpl_utils.uniform_skeleton import uniform_skeleton
from tools.visualize.vis_vec import plot_3d_motion
from utils.smpl_utils.smpl_skeleton import SMPLSkeleton, smplh_chain, smplh_parents, smplh_offsets, smpl_chain, \
    smpl_parents

face_joint_indx = [2, 1, 17, 16]


def smplh_forward(model: SMPLH, body_axis_angle: Union[np.ndarray, torch.Tensor],
                  transl: Union[np.ndarray, torch.Tensor]):
    """
    :param model: SMPLH model
    :param axis_angle: t 22 3. global axis angle
    :param transl: t 3
    :return: positions: t 52 3. whole body joints positions

    """
    if isinstance(body_axis_angle, np.ndarray):
        body_axis_angle = torch.from_numpy(body_axis_angle)
    if isinstance(transl, np.ndarray):
        transl = torch.from_numpy(transl)
    body_axis_angle = rearrange(body_axis_angle, 't j c -> t (j c)')
    global_orient = body_axis_angle[:, :3].to(torch.float32)
    body_pose = body_axis_angle[:, 3:66].to(torch.float32)
    # body_pose = torch.zeros([body_axis_angle.shape[0], 63]).to(torch.float32)
    with torch.no_grad():
        output = model.forward(
            global_orient=global_orient,
            transl=transl,
            body_pose=body_pose).joints
    output = output.cpu().numpy()[:, :52]
    return output


def smpl2smplh(positions_smpl: np.ndarray,
               target_offset: torch.Tensor,
               smplh_skeleton: SMPLSkeleton,
               gender='male'):
    """ convert smpl positions to smplh positions. hand pose are set to same as T-pose
    :param t_pose_positions: joint positions of a t-pose. j 3
    :param target_offset: target offset j 3
    :param gender: gender of the motion performer
    :param positions_smpl: t 24 3 positions
    :param smplh_skeleton: smplh_skeleton
    :return: pos and cont6d of smplh. t 52 3 t
    """

    transl = positions_smpl[:, 0]

    t = positions_smpl.shape[0]
    if gender == 'male':
        smplh = SMPLH(model_path='smpl_models/smplh/SMPLH_MALE.pkl',
                      batch_size=t, gender='male').eval()
        smpl = SMPL_IK(model_path='smpl_models/smpl/basicmodel_m_lbs_10_207_0_v1.1.0.pkl',
                       batch_size=t, gender='male').eval()
    else:
        smplh = SMPLH(model_path='smpl_models/smplh/SMPLH_FEMALE.pkl',
                      batch_size=t, gender='female').eval()
        smpl = SMPL_IK(model_path='smpl_models/smpl/basicmodel_f_lbs_10_207_0_v1.1.0.pkl',
                       batch_size=t, gender='female').eval()
    local_axis_angle_body = smpl.inverse_kinematic_np(
        positions_smpl, smpl_chain, smpl_parents)
    smplh_positions = smplh_forward(smplh, local_axis_angle_body, transl)

    smplh_positions = uniform_skeleton(smplh_positions, target_offset, smplh_skeleton)
    return smplh_positions


if __name__ == '__main__':
    args = argparse.ArgumentParser('test smpl2smplh')
    args.add_argument('--sample_path', default='data/humanml3d/joints_24_52/000001.npy')
    args.add_argument('--reference', default='data/humanml3d/reference.npy')
    args.add_argument('--t_pose', default='data/t_pose_joints.npy')
    args = args.parse_args()

    smplh_skeleton = SMPLSkeleton(torch.from_numpy(np.asarray(smplh_offsets)),
                                  smplh_chain,
                                  smplh_parents)

    reference_data = np.load(args.reference)
    reference_data = torch.from_numpy(reference_data)
    t_pose_positions = np.load(args.t_pose).squeeze()
    tgt_offsets = smplh_skeleton.get_offsets_joints(reference_data[0])

    # t 3
    body_pose = np.load(args.sample_path)
    whole_body_pose = smpl2smplh(body_pose, tgt_offsets, smplh_skeleton)
    basename = os.path.basename(args.sample_path).split('.')[0]
    save_path = f'tmp/{basename}.mp4'
    plot_3d_motion(save_path, smplh_chain, whole_body_pose, 'whole body')
