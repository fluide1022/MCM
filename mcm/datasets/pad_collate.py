"""
    use to pad token sequence of different lengths in a batch to the same
"""
from typing import Sequence, Any, Mapping

import numpy as np
import torch
from mmengine import FUNCTIONS
from mmengine.structures import BaseDataElement
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate as torch_default_collate



@FUNCTIONS.register_module(force=True)
def pad_collate_function(data_batch: Sequence, pad_value: int = 0) -> Any:
    """
        Based on default collate implemented by mmengine.
        Pad token sequence of different lengths in a batch to the same length.

        Convert list of data sampled from dataset into a batch of data, of which
        type consistent with the type of each data_itement in ``data_batch``.

        Different from :func:`pseudo_collate`, ``default_collate`` will stack
        tensor contained in ``data_batch`` into a batched tensor with the
        first dimension batch size, and then move input tensor to the target
        device.

        Different from ``default_collate`` in pytorch, ``default_collate`` will
        not process ``BaseDataElement``.

        This code is referenced from:
        `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.

    :param data_batch: Data sampled from dataset.
    :param pad_value: pad value. if lengths of samples in a batch differ. make sure it's same as your tokenizer
    :return: Any: Data in the same format as the data_itement of ``data_batch``, of which
            tensors have been stacked, and ndarray, int, float have been
            converted to tensors.
    """

    # make sure pad_value is the same as pad_id in your tokenizer!!!!!
    data_item = data_batch[0]
    data_item_type = type(data_item)
    def pad_tensors(tensor_list, batch_first=True):
        # Convert numpy arrays to tensors
        tensor_list = [torch.tensor(item) if isinstance(item, np.ndarray) else item for item in tensor_list]
        padded_sequence = pad_sequence(tensor_list, batch_first=batch_first, padding_value=pad_value)
        return padded_sequence

    if isinstance(data_item, (BaseDataElement, str, bytes)):
        return data_batch
    elif isinstance(data_item, (torch.Tensor, np.ndarray)):
        return pad_tensors(data_batch)
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named_tuple
        return data_item_type(*(pad_collate_function(samples) for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        transposed = list(zip(*data_batch))
        if isinstance(data_item, tuple):
            return [pad_collate_function(samples) for samples in transposed]
        else:
            try:
                return data_item_type([pad_collate_function(samples) for samples in transposed])
            except TypeError:
                return [pad_collate_function(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({key: pad_collate_function([d[key] for d in data_batch]) for key in data_item})

    else:
        return torch_default_collate(data_batch)
