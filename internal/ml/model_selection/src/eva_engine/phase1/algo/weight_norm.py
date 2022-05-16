import torch
from torch import nn
from src.eva_engine.phase1.algo.alg_base import Evaluator
from src.common.constant import Config
from src.eva_engine.phase1.utils.p_utils import get_layer_metric_array


class WeightNormEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor, space_name: str) -> float:
        """
        This is simply sum over all weigth's norm to calculate models performance
        :param arch:
        :param device: CPU or GPU
        :param batch_data:
        :param batch_labels:
        :return:
        """
        grad_norm_arr = get_layer_metric_array(arch, lambda l: l.weight.norm(), mode="param")

        # 3. Sum over all parameter's results to get the final score.
        score = 0.
        for i in range(len(grad_norm_arr)):
            score += grad_norm_arr[i].detach().cpu().numpy().sum()
        return score
