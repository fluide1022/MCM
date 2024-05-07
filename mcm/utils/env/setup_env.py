import platform
from time import time
from typing import Dict

import torch
from mmengine.dist import init_dist, is_distributed, get_dist_info, broadcast
from mmengine.utils.dl_utils import set_multi_processing



def setup_env(env_cfg: Dict, launcher='pytorch') -> None:
    """Setup environment.

    An example of ``env_cfg``::

        env_cfg = dict(
            cudnn_benchmark=True,
            mp_cfg=dict(
                mp_start_method='fork',
                opencv_num_threads=0
            ),
            dist_cfg=dict(backend='nccl', timeout=1800),
            resource_limit=4096
        )

    Args:
        env_cfg (dict): Config for setting environment.
    """
    if env_cfg.get('cudnn_benchmark'):
        torch.backends.cudnn.benchmark = True

    mp_cfg: dict = env_cfg.get('mp_cfg', {})
    set_multi_processing(**mp_cfg, distributed=True)

    # init distributed env first, since logger depends on the dist info.
    if  not is_distributed():
        dist_cfg: dict = env_cfg.get('dist_cfg', {})
        init_dist(launcher, **dist_cfg)

    rank, world_size = get_dist_info()

    timestamp = torch.tensor(time(), dtype=torch.float64)
    # broadcast timestamp from 0 process to other processes
    broadcast(timestamp)


    # https://github.com/pytorch/pytorch/issues/973
    # set resource limit
    if platform.system() != 'Windows':
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        base_soft_limit = rlimit[0]
        hard_limit = rlimit[1]
        soft_limit = min(
            max(env_cfg.get('resource_limit', 4096), base_soft_limit),
            hard_limit)
        resource.setrlimit(resource.RLIMIT_NOFILE,
                           (soft_limit, hard_limit))