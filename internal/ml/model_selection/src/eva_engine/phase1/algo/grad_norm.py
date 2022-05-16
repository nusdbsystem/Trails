from src.eva_engine.phase1.algo.alg_base import Evaluator
from src.eva_engine.phase1.utils.autograd_hacks import *
from src.eva_engine.phase1.utils.p_utils import get_layer_metric_array
from torch import nn
from src.common.constant import Config

class GradNormEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor, space_name: str) -> float:
        """
        This is implementation of paper
        "Keep the Gradients Flowing: Using Gradient Flow to Study Sparse Network Optimization"
        The score takes 5 steps:
            1. Run a forward & backward pass to calculate gradient of loss on weight, grad_w = d_loss/d_w
            2. Then calculate norm for each gradient, grad.norm(p), default p = 2
            3. Sum up all weights' grad norm and get the overall architecture score.
        """

        loss_fn = F.cross_entropy
        grad_norm_arr = []
        # 1. forward on mini-batch
        # logger.info("min-batch is in cuda2 = " + str(batch_data.is_cuda))
        outputs = arch(batch_data)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()

        # 2. lambda function as callback to calculate norm of gradient
        part_grad = get_layer_metric_array(
            arch,
            lambda l:
                l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')

        grad_norm_arr.extend(part_grad)

        # 3. Sum over all parameter's results to get the final score.
        score = 0.
        for i in range(len(grad_norm_arr)):
            score += grad_norm_arr[i].detach().cpu().numpy().sum()
        return score
