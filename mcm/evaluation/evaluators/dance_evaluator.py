import numpy as np
import torch
from ddpm.ddpm_wrapper import DDPMWrapper
from evaluation.functional.m2d.beat_alignment import cal_beat_align
from evaluation.functional.m2d.diversity import cal_diversity
from evaluation.functional.m2d.fid import cal_fid
from evaluation.functional.m2d.pfc import cal_pfc_batch
from evaluation.functional.m2d.m2d_extractor import DanceExtractor
from evaluation.evaluators.base_evaluator import BaseEvaluator
from utils.motion_process import recover_from_ric
from utils.utils import motion_temporal_filter


def normalize(feat, feat2):
    """
    :param feat: bs C
    :param feat2: bs C
    :return:
    """
    # bs
    mean = np.mean(feat, axis=0)
    # bs
    std = np.std(feat, axis=0)
    # bs c, bs c
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)


class DanceEvaluator(BaseEvaluator):
    def __init__(self, opt, ddpm: DDPMWrapper):
        super().__init__(opt, ddpm)
        self.eval_metrics = {
            'num': 0,
            'fid_k': 0.,
            'fid_g': 0.,
            'beat_align': 0,
            'div_k': 0.,
            'div_g': 0.,
            'div_k_gt': 0.,
            'div_g_gt': 0.,
        }
        self.extractor = DanceExtractor()
        self.pred_feature_k = np.empty([0, self.opt.dim_kinemic])
        self.pred_feature_g = np.empty([0, self.opt.dim_manual])
        self.gt_feature_k = np.empty([0, self.opt.dim_kinemic])
        self.gt_feature_g = np.empty([0, self.opt.dim_manual])
        # self.smpl = smplx.create('body_model/smplx/SMPL_MALE.pkl',
        #              model_type="smpl", gender="MALE", ext="pkl",
        #              batch_size=1).to('cuda')

    def final_metric(self):
        self.gt_feature_k, self.pred_feature_k = normalize(self.gt_feature_k, self.pred_feature_k)
        self.gt_feature_g, self.pred_feature_g = normalize(self.gt_feature_g, self.pred_feature_g)
        self.eval_metrics['fid_k'] = cal_fid(
            self.pred_feature_k, self.gt_feature_k
        )
        self.eval_metrics['fid_g'] = cal_fid(
            self.pred_feature_g, self.gt_feature_g
        )
        self.eval_metrics['div_k'] = cal_diversity(self.pred_feature_k)
        self.eval_metrics['div_g'] = cal_diversity(self.pred_feature_g)
        self.eval_metrics['div_k_gt'] = cal_diversity(self.gt_feature_k)
        self.eval_metrics['div_g_gt'] = cal_diversity(self.gt_feature_g)
        self.get_mean_metric()

    def vec2joints(self, vecs: torch.Tensor, m_length: torch.Tensor):
        """ as in bailando, joints are
        :param vecs: b t c
        :param m_length: b
        :return: [t j 3]*b
        """
        joints = []
        for vec, l in zip(vecs, m_length):
            j = recover_from_ric(vec[:l], 22).numpy()  # t j 3
            j = motion_temporal_filter(j, sigma=1)
            # MINS = j.min(axis=0).min(axis=0)
            init_root = j[:1, :1]  # 1,1,3
            # height_floor = j.min(axis=0).min(axis=0)[1]
            # foot at 0
            # j[:, :, 1] -= height_floor
            # j[:, :, 0] -= init_root[:, :, 0]
            # j[:, :, 2] -= init_root[:, :, 2]
            j -= init_root
            # visualize_3d_coordinates(j[0], SMPL_JOINT_NAMES[:22])
            joints.append(j)
        return joints

    def eval_batch(self, pred_motion, batch):
        motion = batch['motion']
        m_length = batch['length']
        # beat feature: B,T
        music_beat = batch['music_beat']
        batch_metric = {key: 0. for key in self.eval_metrics.keys()}
        batch_metric['num'] = len(pred_motion)
        # b t c
        pred_motion = pred_motion.cpu() * self.std + self.mean
        motion = motion.cpu() * self.std + self.mean
        # for i in range(len(pred_motion)):
        #     os.makedirs('test_eval', exist_ok=True)
        #     plot_t2m(pred_motion[i][:m_length[i]].cpu().numpy(), f'test_eval/pred{i}.gif', "", batch['text'][i], 22, False)
        #     plot_t2m(motion[i][:m_length[i]].cpu().numpy(), f'test_eval/gt{i}.gif', "", batch['text'][i], 22, False)
        # [t j c] * b
        pred_motion = self.vec2joints(pred_motion.cpu(), m_length)
        motion = self.vec2joints(motion.cpu(), m_length)
        # b c
        pred_feature_k = self.extractor.extract_kinetic_features_batch(pred_motion)
        pred_feature_g = self.extractor.extract_manual_features_batch(pred_motion)
        gt_feature_k = self.extractor.extract_kinetic_features_batch(motion)
        gt_feature_g = self.extractor.extract_manual_features_batch(motion)
        self.pred_feature_k = np.concatenate([self.pred_feature_k, pred_feature_k], axis=0)
        self.pred_feature_g = np.concatenate([self.pred_feature_g, pred_feature_g], axis=0)
        self.gt_feature_k = np.concatenate([self.gt_feature_k, gt_feature_k], axis=0)
        self.gt_feature_g = np.concatenate([self.gt_feature_g, gt_feature_g], axis=0)

        batch_metric['beat_align'] = cal_beat_align(pred_motion, music_beat, m_length)
        batch_metric['beat_align_gt'] = cal_beat_align(motion, music_beat, m_length)
        batch_metric['pfc'] = cal_pfc_batch(pred_motion)
        batch_metric['pfc_gt'] = cal_pfc_batch(motion)

        return batch_metric

    def get_mean_metric(self):
        self.eval_metrics['beat_align'] /= self.eval_metrics['num']
        self.eval_metrics['beat_align_gt'] /= self.eval_metrics['num']
        self.eval_metrics['pfc'] /= self.eval_metrics['num'] / 10000.
        self.eval_metrics['pfc_gt'] /= self.eval_metrics['num'] / 10000.
