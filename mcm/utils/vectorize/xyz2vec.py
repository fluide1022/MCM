"""
    Transfer smpl joints(22) or smpl-h joints(52) coordinates to 632-dim vectors.
    Borrow from HumanML3D dataset
"""
import argparse
import sys
from multiprocessing.pool import Pool
from os.path import join as pjoin
from typing import Union

import numpy as np
import os

sys.path.append(os.curdir)
from utils.smpl_utils.uniform_skeleton import uniform_skeleton

import torch
from tqdm import tqdm

from utils.geometry.quaternion import qrot_np, qmul_np, qinv_np, qbetween_np
from utils.geometry.rotation_convert import quaternion_to_cont6d_np
from utils.smpl_utils.smpl_skeleton import CHAIN_DICT, smplh_chain, \
    smplh_parents, smpl_chain, smpl_parents, raw_smpl_offsets, raw_smplh_offsets, RAW_OFFSETS_DICT, face_joint_idx, \
    fid_l, fid_r
from utils.smpl_utils.smpl_skeleton import SMPLSkeleton
from utils.vectorize.vec2xyz import recover_from_ric

smpl_skeleton = SMPLSkeleton(raw_smpl_offsets, smpl_chain, smpl_parents)
smplh_skeleton = SMPLSkeleton(raw_smplh_offsets, smplh_chain, smplh_parents)


# positions (batch, joint_num, 3)


def process_file(positions: Union[str, np.ndarray], feet_thre: float, joint_save_path: str, vec_save_path: str,
                 target_offset: torch.Tensor):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''

    if isinstance(positions, str):
        positions = np.load(os.path.join(args.joints_dir, source_file))[:, :args.joints_num]
    assert positions.shape[1] == 22, positions.shape
    positions = uniform_skeleton(positions, target_offset, smpl_skeleton)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_idx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r

    #
    feet_l, feet_r = foot_detect(positions, feet_thre)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        """ move to origin point, rotate to face z+.
        :param positions: origin positions
        :return: new positions
        """
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_cont6d_params(positions, skel: SMPLSkeleton = None):
        if skel is None:
            skel = SMPLSkeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_idx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions, smpl_skeleton)
    positions = get_rifke(positions)

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velocity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    # 4
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    try:
        local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                            global_positions[1:] - global_positions[:-1])
        local_vel = local_vel.reshape(len(local_vel), -1)

        data = np.concatenate([root_data, ric_data[:-1], rot_data[:-1], local_vel, feet_l, feet_r], axis=-1)

        rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), args.joints_num)
        np.save(joint_save_path, rec_ric_data.squeeze().numpy())
        np.save(vec_save_path, data)
        print(vec_save_path, ' has been saved')
    except:
        print(vec_save_path, ' save error')
        return
    return data, global_positions, positions, l_velocity


if __name__ == "__main__":
    args = argparse.ArgumentParser('Convert smpl coordinates to motion vectors')
    args.add_argument('joints_dir', type=str)
    args.add_argument('--new_joints_dir', type=str, default=None)
    args.add_argument('--vec_dir', type=str, default=None)
    args.add_argument('--joints_num', type=int, default=22)
    args.add_argument('--reference_file', type=str, default='data/reference.npy')
    args.add_argument('--skip_exist', action='store_true', help='skip existing files')
    args = args.parse_args()

    # ds_num = 8
    # make dirs
    args.joints_dir = os.path.normpath(args.joints_dir)
    if args.new_joints_dir is None:
        args.new_joints_dir = os.path.join(os.path.dirname(args.joints_dir), 'new_' + os.path.basename(args.joints_dir))
    os.makedirs(args.new_joints_dir, exist_ok=True)
    if args.vec_dir is None:
        args.vec_dir = os.path.join(os.path.dirname(args.joints_dir), 'vecs_' + os.path.basename(args.joints_dir))
    os.makedirs(args.vec_dir, exist_ok=True)

    n_raw_offsets = torch.from_numpy(np.asarray(RAW_OFFSETS_DICT[args.joints_num]))
    kinematic_chain = CHAIN_DICT[args.joints_num]

    # Get offsets of target skeleton
    reference_data = np.load(args.reference_file)
    reference_data = reference_data.reshape(len(reference_data), -1, 3)
    reference_data = torch.from_numpy(reference_data)
    # (joints_num, 3)
    tgt_offsets = smpl_skeleton.get_offsets_joints(reference_data[0])

    source_list = os.listdir(args.joints_dir)
    frame_num = 0
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(args.joints_dir, source_file))[:, :args.joints_num]
        joint_save_path = pjoin(args.new_joints_dir, source_file)
        vec_save_path = pjoin(args.vec_dir, source_file)
        if os.path.exists(joint_save_path) and os.path.exists(vec_save_path) and args.skip_exist:
            continue

        process_file(source_data, 0.002, joint_save_path, vec_save_path, tgt_offsets)

    print(f'Total clips: {len(source_list)}')
