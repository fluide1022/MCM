from typing import Union

import numpy as np
import torch

hands_dim = list(range(4 + 21 * 3, 4 + 51 * 3)) + \
            list(range(4 + 51 * 3 + 21 * 6, 4 + 51 * 3 + 51 * 6)) + \
            list(range(4 + 51 * 3 + 51 * 6 + 21 * 3, 4 + 51 * 3 + 51 * 6 + 51 * 3))

body_dim = list(set(list(range(623))) - set(hands_dim))

hand_joint_mask = [1] * 22 + [0] * 30


def rid_hands(whole_body_vector: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    :param whole_body_vector: 623-dim whole body motion vector.(52-smplh)
    :return: 263-dim body-only vector.(22-smpl)
    """
    assert whole_body_vector.shape[-1] == 623, 'the input should be a 623-dim whole-body vector'
    return whole_body_vector[..., body_dim]


if __name__ == '__main__':
    print(hands_dim)
    print(body_dim)

    whole_body = torch.rand([2, 100, 623])
    print(rid_hands(whole_body).shape)
