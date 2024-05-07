"""
    Visualize the motion vectors
"""
import argparse
import math
import os.path
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

sys.path.append(os.curdir)

from utils.motion_process import recover_from_rot, recover_from_ric
from utils.paramUtil import t2m_raw_offsets, kit_raw_offsets, t2m_kinematic_chain, kit_kinematic_chain
from utils.skeleton import Skeleton


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints: np.ndarray, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()


num_joints_dict = {
    263: 22,
    623: 52,
    251: 21
}

if __name__ == '__main__':
    args = argparse.ArgumentParser('Visualize a motion vector')
    args.add_argument('vec_path', type=str, help='joint path or vector path')
    args.add_argument('--title', type=str, default='some motion')
    args.add_argument('--save_path', type=str, default='tmp')
    args.add_argument('--use_rot', action='store_true')
    args.add_argument('--reference_file', type=str, default='data/humanml3d/reference.npy')
    args.add_argument('--dataset', default='humanml3d')
    args = args.parse_args()
    if not '.' in args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        args.save_path = os.path.join(args.save_path, os.path.basename(args.vec_path).split('.')[0] + '.mp4')
    joints = np.load(args.vec_path)
    raw_offsets = t2m_raw_offsets if args.dataset == 'humanml3d' else kit_raw_offsets
    chain = t2m_kinematic_chain if args.dataset == 'humanml3d' else kit_kinematic_chain
    if joints.shape[-1] != 3:
        # vec2position
        num_joints = num_joints_dict[joints.shape[-1]]
        if args.use_rot:
            skeleton = Skeleton(offset=torch.from_numpy(np.asarray(raw_offsets)),
                                kinematic_tree=chain)
            reference_data = np.load(args.reference_file)
            reference_data = reference_data.reshape(len(reference_data), -1, 3)
            reference_data = torch.from_numpy(reference_data)
            tgt_offsets = skeleton.get_offsets_joints(reference_data[0])

            joints = recover_from_rot(torch.from_numpy(joints[None, ...]).to(torch.float32),
                                      skeleton, joints_num=num_joints).numpy()
            # joints = uniform_skeleton(joints, tgt_offsets, skeleton)
        else:
            joints = recover_from_ric(torch.from_numpy(joints[None, ...]).to(torch.float32),
                                      num_joints_dict[joints.shape[-1]]).numpy()[0]

    else:
        num_joints = joints.shape[-2]
    scale = np.max(joints)
    joints /= scale
    plot_3d_motion(args.save_path, chain, joints, args.title)
