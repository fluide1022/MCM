import torch


def generate_mask(T: int, length: torch.Tensor):
    """
    Args:
        T: pad every motion to T
        length: bs 1. record length for every motion in batch

    Returns:

    """
    B = len(length)
    mask = torch.ones([B, T], dtype=torch.float32)
    for i in range(B):
        l = int(length[i])
        mask[i, :l] = 0
    mask = mask
    return mask


def cross_mask(query_mask: torch.Tensor, key_val_mask: torch.Tensor):
    """
    Args:
        query_mask: [bs len1]
        key_val_mask: [bs len2]

    Returns: attn_mask: bs len1 len2
    """
    query_mask = query_mask.unsqueeze(-1)
    key_val_mask = key_val_mask.unsqueeze(1)
    return query_mask @ key_val_mask
