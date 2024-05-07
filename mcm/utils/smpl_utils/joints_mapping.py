#
from typing import List


def joints_mapping(source_name: List[str], target_name: List[str]):
    """ get index in target_name of names in source_name
    :param source_name: joint name list
    :param target_name: joint name list
    :return:
    """
    index_list = []
    for name in target_name:
        if name in source_name:
            index_list.append(source_name.index(name))
    return index_list