import os
import sys
from glob import glob
from os.path import join

import numpy as np
import torch
from tqdm import tqdm

from mcm.evaluation.functional.m2d.m2d_extractor import M2DExtractor
from mcm.utils.vectorize.vec2xyz import recover_from_ric

sys.path.append(os.curdir)

if __name__=='__main__':
    m2d_extractor= M2DExtractor()
    all_feature_k=[]
    all_feature_g=[]
    vec_root = 'data/aist_plusplus_final/vecs_joints_22'
    for vec_path in tqdm(glob(join(vec_root, '*.npy'))):
        vec = np.load(vec_path)
        joints = recover_from_ric(torch.from_numpy(vec[None]), joints_num=22).cpu().numpy()[0]
        feature_k = m2d_extractor.extract_kinetic_features(joints)
        feature_g = m2d_extractor.extract_manual_features(joints)
        all_feature_k.append(feature_k)
        all_feature_g.append(feature_g)
    all_feature_k=np.stack(all_feature_k, axis=0)
    all_feature_g=np.stack(all_feature_g, axis=0)
    mean_k = all_feature_k.mean(axis=0)
    std_k = all_feature_k.std(axis=0)
    mean_g = all_feature_g.mean(axis=0)
    std_g = all_feature_g.std(axis=0)
    np.save('data/aist_plusplus_final/all_feature_k.npy', all_feature_k)
    np.save('data/aist_plusplus_final/all_feature_g.npy', all_feature_g)
    np.save('data/aist_plusplus_final/mean_k.npy', mean_k)
    np.save('data/aist_plusplus_final/std_k.npy', std_k)
    np.save('data/aist_plusplus_final/mean_g.npy', mean_g)
    np.save('data/aist_plusplus_final/std_g.npy', std_g)
