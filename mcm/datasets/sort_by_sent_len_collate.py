from typing import Sequence, Any, Mapping

from mmengine import FUNCTIONS
from mmengine.structures import BaseDataElement
from torch.utils.data._utils.collate import default_collate as torch_default_collate
from mmengine.dataset.utils import default_collate

@FUNCTIONS.register_module(force=True)
def sort_by_sent_len_collate_function(data_batch: Sequence) -> Any:
    data_item = data_batch[0]
    data_item_type = type(data_item)
    data_batch.sort(key=lambda x: x['sent_len'], reverse=True)

    if isinstance(data_item, (BaseDataElement, str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named_tuple
        return data_item_type(*(default_collate(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [default_collate(samples)
                    for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type(
                    [default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [default_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({
            key: default_collate([d[key] for d in data_batch])
            for key in data_item
        })
    else:
        return torch_default_collate(data_batch)
