# transfer origin std to modified std, though i don't know why.
import argparse

import numpy as np

FEAT_BIAS = 25
if __name__ == '__main__':
    args = argparse.ArgumentParser('modify Std')
    args.add_argument('ori_std', type=str, help='original std.npy file path')
    args.add_argument('--num_joints', default=21, type=int, help='num joints')
    args = args.parse_args()
    std = np.load(args.ori_std)
    # root_rot_velocity (B, seq_len, 1)
    std[0:1] = std[0:1] / FEAT_BIAS
    # root_linear_velocity (B, seq_len, 2)
    std[1:3] = std[1:3] / FEAT_BIAS
    # root_y (B, seq_len, 1)
    std[3:4] = std[3:4] / FEAT_BIAS
    # ric_data (B, seq_len, (joint_num - 1)*3)
    std[4: 4 + (args.num_joints - 1) * 3] = std[4: 4 + (args.num_joints - 1) * 3] / 1.0
    # rot_data (B, seq_len, (joint_num - 1)*6)
    std[4 + (args.num_joints - 1) * 3: 4 + (args.num_joints - 1) * 9] = std[4 + (args.num_joints - 1) * 3: 4 + (
            args.num_joints - 1) * 9] / 1.0
    # local_velocity (B, seq_len, joint_num*3)
    std[4 + (args.num_joints - 1) * 9: 4 + (args.num_joints - 1) * 9 + args.num_joints * 3] = std[
                                                                               4 + (args.num_joints - 1) * 9: 4 + (
                                                                                       args.num_joints - 1) * 9 + args.num_joints * 3] / 1.0
    # foot contact (B, seq_len, 4)
    std[4 + (args.num_joints - 1) * 9 + args.num_joints * 3:] = std[
                                                      4 + (args.num_joints - 1) * 9 + args.num_joints * 3:] / FEAT_BIAS

    np.save(args.ori_std.replace('Std.npy', 'modified_std.npy'), std)
