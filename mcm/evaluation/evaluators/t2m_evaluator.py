from typing import List

import torch
from einops import rearrange
from mmengine import MODELS
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm.ddpm_wrapper import DDPMWrapper
from evaluation.evaluators.base_evaluator import BaseEvaluator
from evaluation.functional.t2m.diversity import cal_diversity
from evaluation.functional.t2m.fid import cal_fid
from evaluation.functional.t2m.matching_score_precision import cal_matching_score_r_precision
from evaluation.functional.t2m.multimodality import cal_multimodality


class T2MEvaluator(BaseEvaluator):
    def __init__(self, cfg, ddpm: DDPMWrapper):
        """
        :param cfg: eval_cfg
        :param ddpm: ddpm wrapper
        """
        super().__init__(cfg, ddpm)
        self.extractor = MODELS.build(cfg)
        self.eval_metrics = {
            'num': 0,
            'fid': 0.,
            'matching_score': 0,
            'diversity': 0.,
            'multi_modality': 0.,
        }
        self.pred_motion_embs = torch.empty([0, self.cfg.dim_coemb_hidden])
        self.gt_motion_embs = torch.empty([0, self.cfg.dim_coemb_hidden])
        self.m_lens = torch.empty([0])
        # for multimodality evaluation
        self.mm_motion_embs = torch.empty([0, self.cfg.mm_num_repeats, self.cfg.dim_coemb_hidden])
        self.mm_lens = torch.empty([0])

    def eval(self, test_loader: DataLoader):
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='generation for fid,diversity, matching score ...')):
            # pred_motion = self.ddpm.generate(
            #     m_lens=batch['m_length'],
            #     dim_pose=self.cfg['dim_pose'],
            #     text=batch['caption'],
            #     batch_size=len(batch['motion'])
            # )
            # b t c
            # pred_motion = torch.stack(pred_motion, dim=0)
            # batch_metrics = self.eval_batch(pred_motion, batch)
            batch_metrics = self.eval_batch(None, batch)
            self.update_metric(batch_metrics)

        self.final_metric()

    def eval_mm(self, mm_test_loader: DataLoader):
        assert mm_test_loader.batch_size == self.cfg.mm_num_samples
        all_mm_embeddings=[]
        for repeat in tqdm(range(self.cfg['mm_num_repeats']), desc='evaluating mm'):
            for mm_samples in mm_test_loader:
                motion: Tensor = mm_samples['motion']
                m_length: Tensor = mm_samples['m_length']
                caption: List[str] = mm_samples['caption']

                pred_motion = self.ddpm.generate(m_lens=m_length,
                                            dim_pose=self.cfg['dim_pose'],
                                            batch_size=len(motion),
                                            text=caption)

                pred_motion = torch.stack(pred_motion, dim=0)
                motion_embedding = self.extractor.get_motion_embeddings(
                    pred_motion, m_lens=m_length)
                all_mm_embeddings.append(motion_embedding)
                break
        # [mm_repeat, mm_num_samples, dim_pose]
        all_mm_embeddings = torch.stack(all_mm_embeddings)
        all_mm_embeddings = rearrange(all_mm_embeddings, 'r n c -> n r c')
        self.eval_metrics['multi_modality'] = cal_multimodality(all_mm_embeddings, self.cfg['mm_num_times'])

    def tensor2numpy(self):
        self.mm_motion_embs = self.mm_motion_embs.cpu().numpy()
        self.pred_motion_embs = self.pred_motion_embs.cpu().numpy()
        self.gt_motion_embs = self.gt_motion_embs.cpu().numpy()
        self.m_lens = self.m_lens.cpu().numpy()
        self.mm_lens = self.mm_lens.cpu().numpy()

    def final_metric(self):
        self.tensor2numpy()
        # fid diversity multimodality cannot be calculated batch by batch.  matching score and r precision is ok
        # fid diversity multi_modality no need to cal mean
        # self.eval_metrics['fid'] = cal_fid(self.pred_motion_embs,
        #                                    self.gt_motion_embs)
        # self.eval_metrics['diversity'] = cal_diversity(pred_motion=self.pred_motion_embs,
        #                                                    diversity_times=self.cfg.diversity_times)
        self.eval_metrics['gt_diversity'] = cal_diversity(pred_motion=self.gt_motion_embs,
                                                              diversity_times=self.cfg.diversity_times)
        self.get_mean_metric()

    def get_mean_metric(self):
        # mean
        for key, value in self.eval_metrics.items():
            if key.startswith('matching_score') or key.startswith('r_precision'):
                self.eval_metrics[key] = value / self.eval_metrics['num']

    def eval_batch(self, pred_motion, batch):
        batch_metric = {key: 0. for key in self.eval_metrics.keys()}
        batch_metric['num'] = len(batch['motion'])
        # TODO
        # eval other metric for specified scenario
        # text_emb, pred_motion_emb = self.extractor.get_co_embeddings(
        #     motions=pred_motion,
        #     word_embs=batch['word_embeddings'],
        #     pos_ohot=batch['pos_one_hots'],
        #     cap_lens=batch['sent_len'],
        #     m_lens=batch['m_length']
        # )
        #
        # batch_metric['matching_score'], r_precision = \
        #     cal_matching_score_r_precision(
        #         motion_embeddings=pred_motion_emb.cpu().numpy(),
        #         text_embeddings=text_emb.cpu().numpy(),
        #     )
        # for i in range(len(r_precision)):
        #     batch_metric[f'r_precision_top{i + 1}'] = r_precision[i]
        #
        # self.pred_motion_embs = torch.cat([self.pred_motion_embs, pred_motion_emb.to(self.pred_motion_embs.device)],
        #                                   dim=0)

        text_emb, gt_motion_emb = self.extractor.get_co_embeddings(
            motions=batch['motion'],
            word_embs=batch['word_embeddings'],
            pos_ohot=batch['pos_one_hots'],
            cap_lens=batch['sent_len'],
            m_lens=batch['m_length']
        )

        gt_motion_emb = self.extractor.get_motion_embeddings(
            motions=batch['motion'],
            m_lens=batch['m_length']
        )
        print(gt_motion_emb.shape)
        batch_metric['matching_score_gt'], r_precision = \
            cal_matching_score_r_precision(
                motion_embeddings=gt_motion_emb.cpu().numpy(),
                text_embeddings=text_emb.cpu().numpy(),
            )
        for i in range(len(r_precision)):
            batch_metric[f'r_precision_gt_top{i + 1}'] = r_precision[i]

        self.gt_motion_embs = torch.cat([self.gt_motion_embs, gt_motion_emb.to(self.gt_motion_embs.device)], dim=0)
        self.m_lens = torch.cat([self.m_lens, batch['m_length'].to(self.m_lens.device)], dim=0)
        return batch_metric
