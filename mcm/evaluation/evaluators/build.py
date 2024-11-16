from addict import Dict

from ddpm.ddpm_wrapper import DDPMWrapper
from evaluation.evaluators.dance_evaluator import DanceEvaluator
from evaluation.evaluators.t2m_evaluator import T2MEvaluator

evaluator_dict = {
    't2m': T2MEvaluator,
    'dance': DanceEvaluator
}


def build_evaluator(task: str,
                    opt: Dict,
                    ddpm: DDPMWrapper):
    assert task in ['t2m', 'dance'], f'invalid scene {task}'
    return evaluator_dict[task](opt, ddpm)
