from collections import defaultdict
from typing import Optional, Sequence, Dict, Tuple

import numpy as np
import torch
from mmengine import MMLogger
from mmengine.evaluator.evaluator import BaseMetric
from tqdm import tqdm

from evaluation.functional.keypoint_eval import keypoint_mpjpe
from mmengine.registry import METRICS

@METRICS.register_module(force=True)
class MotionVQVAEMetric(BaseMetric):
    """MotionVQVAE evaluation metric.

    Calculate the mean per-joint position error (MPJPE) of keypoints.

    Note:
        - length of dataset: N
        - num_keypoints: K
        - number of keypoint dimensions: D (typically D = 2)

    Args:
        mode (str): Method to align the prediction with the
            ground truth. Supported configs are:

                - ``'mpjpe'``: no alignment will be applied
                - ``'p-mpjpe'``: align in the least-square sense in scale
                - ``'n-mpjpe'``: align in the least-square sense in
                    scale, rotation, and translation.

        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.
    """

    ALIGNMENT = {'mpjpe': 'none', 'p-mpjpe': 'procrustes', 'n-mpjpe': 'scale'}

    def __init__(self,
                 mode: str = 'mpjpe',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        allowed_modes = self.ALIGNMENT.keys()
        if mode not in allowed_modes:
            raise KeyError("`mode` should be 'mpjpe', 'p-mpjpe', or "
                           f"'n-mpjpe', but got '{mode}'.")
        self.mode = mode
        self.align = self.ALIGNMENT[mode]
        self.hands_dataset = ['motionx', 'zhijiang_sign', 'beat']

    def process(self,
                data_batch: Dict,
                data_samples: Sequence[Dict]) -> None:
        """
        Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        :param data_batch:  A batch of data from the validation dataloader.
        :param data_samples:  A batch of outputs from the model.val_step().
        :return:
        """
        self.results = self.results + list(data_samples)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are the corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        # pred_coords: [N, T, J, C]
        pred_coords = np.stack([result['pred_coord'] for result in results], axis=0)
        # gt_coords: [N, T, J, C]
        gt_coords = np.stack([result['gt_coord'] for result in results], axis=0)
        # mask: [N,J]
        mask = np.stack([result['mask'] for result in results], axis=0)
        assert len(mask.shape) == 2, mask.shape
        # record sample indices for each dataset
        dataset_indices = defaultdict(list)

        for idx, result in tqdm(enumerate(results), desc='Evaluating all datasets'):
            dataset = result['dataset']
            dataset_indices[dataset].append(idx)

        error_name = self.mode.upper()

        logger.info(f'Evaluating {self.mode.upper()}...')
        metrics = dict()

        metrics[error_name] = keypoint_mpjpe(pred_coords, gt_coords, mask,
                                             self.ALIGNMENT[self.mode])

        for dataset, indices in tqdm(dataset_indices.items(), desc='Evaluating specific datasets'):

            metrics[f'{error_name}_{dataset}'] = \
                keypoint_mpjpe(pred_coords[indices], gt_coords[indices], mask[indices], self.align)
            # hands only
            if dataset in self.hands_dataset:
                metrics[f'{error_name}_hands_{dataset}'] = keypoint_mpjpe(
                    pred_coords[indices, :, 22:], gt_coords[indices, :, 22:], alignment=self.align)
        return metrics


if __name__ == '__main__':
    mpjpe = METRICS.build()
