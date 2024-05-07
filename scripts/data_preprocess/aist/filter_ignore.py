from typing import List


def filter_index_list(path: str, ignore_list: List[str]):
    with open(path, 'r') as fp:
        index_list = fp.readlines()
    index_list = [item.strip() for item in index_list]

    def filter_fn(item):
        return not item in ignore_list
    print(f'filter before {len(index_list)}')
    index_list = list(filter(filter_fn, index_list))
    print(f'filter after {len(index_list)}')
    with open(path,'w') as fp:
        fp.write('\n'.join(index_list))

if __name__ == '__main__':
    with open('./data/aist_plusplus_final/ignore_list.txt', 'r') as fp:
        ignore_list = fp.readlines()
    ignore_list = [item.strip() for item in ignore_list]
    filter_list = [
        'data/aist_plusplus_final/test.txt',
        'data/aist_plusplus_final/train.txt',
        'data/aist_plusplus_final/val.txt',
        'data/aist_plusplus_final/train_val.txt',
    ]
    for path in filter_list:
        filter_index_list(path,ignore_list)
