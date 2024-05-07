import glob
import os.path

from tqdm import tqdm

if __name__ == '__main__':
    vec_root = 'data/beat/vecs_joints_22'
    idx_list = []
    for path in tqdm(glob.glob(os.path.join(vec_root, '*.npy'))):
        idx_list.append(os.path.basename(path).split('.')[0])
    with open('data/beat/all.txt', 'w') as fp:
        fp.write('\n'.join(idx_list))
