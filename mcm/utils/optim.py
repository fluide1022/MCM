import copy
from typing import Union

from mmengine import OPTIM_WRAPPER_CONSTRUCTORS, Config, ConfigDict
from mmengine.device import is_npu_support_full_precision, is_npu_available
from mmengine.optim import OptimWrapper
from torch import nn


def build_optim_wrapper(model: nn.Module,
                        cfg: Union[dict, Config, ConfigDict]) -> OptimWrapper:
    """Build function of OptimWrapper.

    If ``constructor`` is set in the ``cfg``, this method will build an
    optimizer wrapper constructor, and use optimizer wrapper constructor to
    build the optimizer wrapper. If ``constructor`` is not set, the
    ``DefaultOptimWrapperConstructor`` will be used by default.

    Args:
        model (nn.Module): Model to be optimized.
        cfg (dict): Config of optimizer wrapper, optimizer constructor and
            optimizer.

    Returns:
        OptimWrapper: The built optimizer wrapper.
    """
    optim_wrapper_cfg = copy.deepcopy(cfg)
    constructor_type = optim_wrapper_cfg.pop('constructor',
                                             'DefaultOptimWrapperConstructor')
    paramwise_cfg = optim_wrapper_cfg.pop('paramwise_cfg', None)

    # Since the current generation of NPU(Ascend 910) only supports
    # mixed precision training, here we turn on mixed precision
    # to make the training normal
    if is_npu_available() and not is_npu_support_full_precision():
        optim_wrapper_cfg['type'] = 'AmpOptimWrapper'

    optim_wrapper_constructor = OPTIM_WRAPPER_CONSTRUCTORS.build(
        dict(
            type=constructor_type,
            optim_wrapper_cfg=optim_wrapper_cfg,
            paramwise_cfg=paramwise_cfg))
    optim_wrapper = optim_wrapper_constructor(model)
    return optim_wrapper