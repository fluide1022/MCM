import os
from typing import List


def in_list(idx: str, filter_list: List[str]):
    for name in filter_list:
        if name in idx and len(name) > 1:
            return True
    return False


if __name__ == '__main__':
    with open('./data/aist_plusplus_final/ignore_list.txt', 'r') as fp:
        ignore_list = fp.readlines()
    ignore_list = [item.strip() for item in ignore_list]
    with open('./data/aist_plusplus_final/test.txt', 'r') as fp:
        test_list = fp.readlines()
    with open('./data/aist_plusplus_final/val.txt', 'r') as fp:
        val_list = fp.readlines()
    test_list = [item.strip() for item in test_list]+[item.strip() for item in val_list]
    train_val_list = []
    for file in os.listdir('data/aist_plusplus_final/joints_22_vecs/'):
        idx = file[:-4]
        if in_list(idx, ignore_list) or in_list(idx, test_list):
            continue
        train_val_list.append(idx)
    with open('./data/aist_plusplus_final/train.txt', 'w') as fp:
        fp.write('\n'.join(train_val_list))
    print(len(train_val_list))
