from typing import Union

import numpy as np
import torch

def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def cal_matching_score_r_precision(motion_embeddings: Union[torch.Tensor, np.ndarray],
                                   text_embeddings: Union[torch.Tensor, np.ndarray],
                                   reduce=False):
    """  when calculating precision, we calculate match rate between the motion and its caption
    in the 32-samples batch.
    "match" means having the lowest topk dist in the batch.
    :param text_embeddings: all text embeddings extracted from humanml3d
    :param motion_embeddings: all motion embeddings extracted from humanml3d
    :return: matching score and r_precision,
        note that the true r_precision is r_precision/num_samples
    """
    if isinstance(motion_embeddings, torch.Tensor):
        motion_embeddings = motion_embeddings.numpy()
        text_embeddings = text_embeddings.numpy()

    motion_embeddings = motion_embeddings.copy()
    text_embeddings = text_embeddings.copy()
    batch_size = 32

    matching_score = 0.
    top_k_count = 0
    shuffle_idx = torch.randperm(len(motion_embeddings))
    motion_embeddings = motion_embeddings[shuffle_idx]
    text_embeddings = text_embeddings[shuffle_idx]
    num_samples = len(motion_embeddings) // batch_size * batch_size
    for idx in range(0, num_samples, batch_size):
        # bs bs, distance from every text to every motion
        dist_mat = euclidean_distance_matrix(
            text_embeddings[idx:idx + batch_size], motion_embeddings[idx:idx + batch_size])
        # matching score between every text to corresponding motion
        matching_score += dist_mat.trace()
        # a mapping for every text_embedding to motion embeddings,
        argsmax = np.argsort(dist_mat, axis=1)

        top_k_mat = calculate_top_k(argsmax, top_k=3)
        #
        top_k_count += top_k_mat.sum(axis=0)
    if reduce:
        top_k_count = top_k_count / num_samples
        matching_score = matching_score / num_samples
    return matching_score, top_k_count


def calculate_top_k(mat, top_k: int):
    # calculate if the correct text embedding has the highest topk score in batch?
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat
