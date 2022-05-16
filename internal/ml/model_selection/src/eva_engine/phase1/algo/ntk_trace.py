import time

import numpy as np
from src.eva_engine.phase1.algo.alg_base import Evaluator
from src.common.constant import Config
from src.eva_engine.phase1.utils.autograd_hacks import *


class NTKTraceEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, arch: nn.Module, device, batch_data: object, batch_labels: torch.Tensor, space_name: str) -> float:
        """
        This is implementation of paper
        "NASI: Label- and Data-agnostic Neural Architecture Search at Initialization"
        The score takes 5 steps:
            1. run forward on a mini-batch
            2. output = sum( [ yi for i in mini-batch N ] ) and then run backward
            3. explicitly calculate gradient of f on each example, df/dxi,
                grads = [ df/ dxi for xi in [1, ..., N] ], dim = [N, number of parameters]
            4. calculate NTK = grads * grads_t
            5. calculate M_trace = traceNorm(NTK), score = np.sqrt(trace_norm / batch_size)
        """

        # this is for the structure data,
        if space_name == Config.MLPSP:
            batch_size = batch_data["value"].shape[0]
        else:
            batch_size = batch_data.shape[0]

        add_hooks(arch)

        # 1. forward on mini-batch
        outputs = arch.forward(batch_data)

        # 2. run backward
        # todo: why sum all sample's output ?
        output_f = sum(outputs[torch.arange(batch_size), batch_labels])
        output_f.backward()

        # 3. calculate gradient for each sample in the batch
        # grads = ∇0 f(X), it is N*P , N is number of sample, P is number of parameters,
        compute_grad1(arch, loss_type='sum')

        grads = [param.grad1.flatten(start_dim=1) for param in arch.parameters() if hasattr(param, 'grad1')]

        # remove those in GPU
        del arch
        torch.cuda.empty_cache()

        # print("gradient calculated done, delete arch, begin to compute NTK")

        # 4. ntk = ∇0 f(X) * Transpose( ∇0 f(X) ) [ batch_size * batch_size ]
        begin = time.time()
        grads_final = torch.zeros(batch_size, batch_size).to(device)
        for ele in grads:
            grads_final += torch.matmul(ele, ele.t())
        end = time.time()

        ntk = grads_final.detach()
        del grads
        del grads_final
        torch.cuda.empty_cache()

        # 5. calculate M_trace = sqrt ( |ntk|_tr * 1/m )

        # For a Hermitian matrix, like a density matrix,
        # the absolute value of the eigenvalues are exactly the singular values,
        # so the trace norm is the sum of the absolute value of the eigenvalues of the density matrix.
        # eigenvalues, _ = torch.symeig(ntk)  # ascending
        eigenvalues, _ = torch.linalg.eigh(ntk)

        trace_norm = eigenvalues.cpu().numpy().sum()
        score = np.sqrt(trace_norm / batch_size)

        del eigenvalues
        del ntk
        torch.cuda.empty_cache()
        return score
