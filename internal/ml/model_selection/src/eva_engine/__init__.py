from src.common.constant import *
from src.eva_engine.phase1.algo.grad_norm import GradNormEvaluator
from src.eva_engine.phase1.algo.grad_plain import GradPlainEvaluator
from src.eva_engine.phase1.algo.nas_wot import NWTEvaluator
from src.eva_engine.phase1.algo.ntk_condition_num import NTKCondNumEvaluator
from src.eva_engine.phase1.algo.ntk_trace import NTKTraceEvaluator
from src.eva_engine.phase1.algo.ntk_trace_approx import NTKTraceApproxEvaluator
from src.eva_engine.phase1.algo.prune_fisher import FisherEvaluator
from src.eva_engine.phase1.algo.prune_grasp import GraspEvaluator
from src.eva_engine.phase1.algo.prune_snip import SnipEvaluator
from src.eva_engine.phase1.algo.prune_synflow import SynFlowEvaluator
from src.eva_engine.phase1.algo.express_flow import ExpressFlowEvaluator
from src.eva_engine.phase1.algo.weight_norm import WeightNormEvaluator
from src.eva_engine.phase1.algo.knas import KNASEvaluator

# evaluator mapper to register many existing evaluation algorithms
evaluator_register = {

    CommonVars.ExpressFlow: ExpressFlowEvaluator(),

    # # sum on gradient
    CommonVars.GRAD_NORM: GradNormEvaluator(),
    CommonVars.GRAD_PLAIN: GradPlainEvaluator(),
    #
    # # training free matrix
    # CommonVars.JACOB_CONV: JacobConvEvaluator(),
    CommonVars.NAS_WOT: NWTEvaluator(),

    # this is ntk based
    CommonVars.NTK_CONDNUM: NTKCondNumEvaluator(),
    CommonVars.NTK_TRACE: NTKTraceEvaluator(),

    CommonVars.NTK_TRACE_APPROX: NTKTraceApproxEvaluator(),

    # # prune based
    CommonVars.PRUNE_FISHER: FisherEvaluator(),
    CommonVars.PRUNE_GRASP: GraspEvaluator(),
    CommonVars.PRUNE_SNIP: SnipEvaluator(),
    CommonVars.PRUNE_SYNFLOW: SynFlowEvaluator(),

    # # sum of weight
    CommonVars.WEIGHT_NORM: WeightNormEvaluator(),

    CommonVars.KNAS: KNASEvaluator(),

}

