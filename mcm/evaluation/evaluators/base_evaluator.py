import torch
from tqdm import tqdm

from ddpm.ddpm_wrapper import DDPMWrapper


class BaseEvaluator(object):
    def __init__(self, cfg, ddpm: DDPMWrapper):
        self.eval_metrics = {}
        self.ddpm = ddpm
        self.cfg = cfg

    def eval(self, test_loader):
        self.mean = torch.Tensor(test_loader.dataset.mean)
        self.std = torch.Tensor(test_loader.dataset.std)
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='generating for evaluation')):
            pred_motion = self.ddpm.generate(
                m_lens=batch['length'],
                dim_pose=self.cfg['dim_pose'],
                text=batch['caption'],
                audio=batch.get('audio'),
                batch_size=len(batch['motion'])
            )
            # b t c
            pred_motion = torch.stack(pred_motion, dim=0)
            batch_metrics = self.eval_batch(pred_motion, batch)
            self.update_metric(batch_metrics)
        self.final_metric()


    def eval_batch(self, pred_motion, batch):
        batch_metric = {key: 0. for key in self.eval_metrics.keys()}
        batch_metric['num'] = len(batch)
        # TODO
        # eval other metric for specified scenario
        return batch_metric

    def update_metric(self, batch_metric):
        for key, value in batch_metric.items():
            if self.eval_metrics.get(key) is not None:
                self.eval_metrics[key] += value
            else:
                self.eval_metrics[key] = value

    def final_metric(self):
        self.get_mean_metric()

    def get_mean_metric(self):
        # mean
        for key, value in self.eval_metrics.items():
            if key != 'num':
                self.eval_metrics[key] = value / self.eval_metrics['num']
