import os

from tqdm import tqdm

vec_root = 'data/aist_plusplus_final/joints_22_vecs'


def get_sliced_index(ori_index_path: str, new_index_path: str):
    new_index_list = []
    with open(ori_index_path, 'r') as fp:
        ori_index_list = fp.readlines()
    for index in tqdm(ori_index_list):
        index = index.strip()
        slice_id = 0
        new_index = f'{index}_slice{str(slice_id)}'
        new_path = os.path.join(vec_root, f'{new_index}.npy')
        mirror_index = f'M{new_index}.npy'
        mirror_new_path = os.path.join(vec_root, mirror_index)
        if slice_id == 0:
            assert os.path.exists(new_path)
        while os.path.exists(new_path):
            new_index_list.append(new_index)
            assert os.path.exists(mirror_new_path),new_index
            new_index_list.append(mirror_new_path)
            slice_id += 1
            new_index = f'{index}_slice{str(slice_id)}'
            new_path = os.path.join(vec_root, f'{new_index}.npy')
            mirror_index = f'M{new_index}.npy'
            mirror_new_path = os.path.join(vec_root, mirror_index)
    with open(new_index_path, 'w') as fp:
        fp.write('\n'.join(new_index_list))
    print(len(new_index_list))


if __name__ == '__main__':
    all_path = 'data/aist_plusplus_final/splits/all_unslice.txt'
    new_all_path = 'data/aist_plusplus_final/all.txt'
    test_path = 'data/aist_plusplus_final/splits/test_unslice.txt'
    new_test_path = 'data/aist_plusplus_final/test.txt'
    val_path = 'data/aist_plusplus_final/splits/val_unslice.txt'
    new_val_path = 'data/aist_plusplus_final/val.txt'
    train_path = 'data/aist_plusplus_final/splits/train_unslice.txt'
    new_train_path = 'data/aist_plusplus_final/train.txt'
    train_val_path = 'data/aist_plusplus_final/splits/train_val_unslice.txt'
    new_train_val_path = 'data/aist_plusplus_final/train_val.txt'
    get_sliced_index(all_path, new_all_path)
    get_sliced_index(test_path, new_test_path)
    get_sliced_index(val_path, new_val_path)
    get_sliced_index(train_path, new_train_path)
    get_sliced_index(train_val_path, new_train_val_path)
