import logging
import math
import os.path
import time
from typing import Dict, Union

import os.path as osp

import torch
from mmengine import MMLogger


def log_current_loss(start_time, niter_state, losses, epoch=None, inner_iter=None, logger: logging.Logger = None):
    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        if logger is None:
            print('epoch: %3d niter: %6d  inner_iter: %4d' % (epoch, niter_state, inner_iter), end=" ")
        else:
            logger.info('epoch: %3d niter: %6d  inner_iter: %4d' % (epoch, niter_state, inner_iter))
    now = time.time()
    message = '%s' % (as_minutes(now - start_time))

    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    if logger is None:
        print(message)
    else:
        logger.info(message)


def build_logger(cfg,
                 log_level: Union[int, str] = 'INFO',
                 log_file: str = None,
                 **kwargs) -> MMLogger:
    """Build a global asscessable MMLogger.

    Args:
        log_level (int or str): The log level of MMLogger handlers.
            Defaults to 'INFO'.
        log_file (str, optional): Path of filename to save log.
            Defaults to None.
        **kwargs: Remaining parameters passed to ``MMLogger``.

    Returns:
        MMLogger: A MMLogger object build from ``logger``.
    """
    timestamp = torch.tensor(time.time(), dtype=torch.float64)
    timestamp = time.strftime('%Y%m%d_%H%M%S',
                              time.localtime(timestamp.item()))
    log_dir = osp.join(cfg.work_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    if log_file is None:
        log_file = osp.join(log_dir, f'{timestamp}.log')

    log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
    log_cfg.setdefault('name', timestamp)
    # `torch.compile` in PyTorch 2.0 could close all user defined handlers
    # unexpectedly. Using file mode 'a' can help prevent abnormal
    # termination of the FileHandler and ensure that the log file could
    # be continuously updated during the lifespan of the runner.
    log_cfg.setdefault('file_mode', 'a')

    return MMLogger.get_instance(**log_cfg)  # type: ignore


def log_resume(epoch, it, logger: logging.Logger = None):
    message = f'resume from epoch {epoch}, iteration {it}'
    if logger is None:
        print(message)
    else:
        logger.info(message)


def log_evaluation(eval_metric: Dict, logger: logging.Logger = None):
    for key, value in eval_metric.items():
        if logger is None:
            print(f'{key}: {value:.4f}')
        else:
            logger.info(f'{key}: {value:.4f}')
