import os
import sys
from collections import defaultdict
from functools import partial
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine import METRICS
from mmengine.evaluator.evaluator import BaseMetric

sys.path.append(os.curdir)
from evaluation.functional.accuracy import accuracy_score


@METRICS.register_module(force=True)
class TokenMetric(BaseMetric):
    """
    TODO: implement a metric
    """
    default_prefix: Optional[str] = 'metric'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # like when calculating loss, shift should be operated
        # make sure ignore_idx is same as the padding idx
        self.token_evaluators = dict(
            accuracy=partial(accuracy_score, normalize=True, shift=True, ignore_idx=0)
        )

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self,
                data_batch: Dict,
                data_samples: Sequence[Dict]) -> None:
        """
        Process one batch of data samples and pred_ids. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        :param data_batch:  A batch of data from the validation dataloader.
        :param data_samples:  A batch of outputs from the model.val_step().
        :return: None
        """

        self.results = self.results + list(data_samples)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        metrics = dict()
        task_indices = defaultdict(list)
        dataset_indices = defaultdict(list)
        dataset_task_indices = defaultdict(list)

        for idx, result in enumerate(results):
            task = result['task']
            task_indices[task].append(idx)

            dataset = result['dataset']
            dataset_indices[dataset].append(idx)

            # for dataset and task specific evaluation
            dataset_task_indices[f'{dataset}_{task}'].append(idx)

        pred_ids = [result['pred_ids'] for result in results]
        labels = [result['labels'] for result in results]
        # token metrics, like accuracy
        for metric, evaluator in self.token_evaluators.items():
            metrics[metric] = evaluator(
                np.concatenate(labels, axis=0),
                np.concatenate(pred_ids, axis=0)
           )

        for task, indices in task_indices.items():
            for metric, evaluator in self.token_evaluators.items():
                task_pred_ids = [pred_ids[i] for i in indices]
                task_labels = [labels[i] for i in indices]
                metrics[f'{metric}_{task}'] = evaluator(
                    np.concatenate(task_labels, axis=0),
                    np.concatenate(task_pred_ids, axis=0),
                )

        return metrics


if __name__ == '__main__':
    metric = TokenMetric()
    results = [
        dict(
            dataset='humanml3d',
            task='t2m',
            pred_ids=np.random.randint(0, 35498, size=[200]),
            labels=np.random.randint(0, 35498, size=[200]),
        ),
        dict(
            dataset='humanml3d',
            task='t2m',
            pred_ids=np.random.randint(0, 35498, size=[150]),
            labels=np.random.randint(0, 35498, size=[150]),
        ),
    ]

    print(metric.compute_metrics(results))
