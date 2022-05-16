import math
import os
import time
import random
from exps.shared_args import parse_arguments
import calendar
from src.common.constant import Config
from src.tools.compute import log_scale_x_array
import traceback


def run_one_fixed_budget_alg(sh, time_per_epoch):
    # calculate min time required for evaluating 500 models
    min_epoch_for_fixed_k = sh.pre_calculate_epoch_required(K=total_models, U=1)
    # in second

    min_array = log_scale_x_array(num_points=args.num_points, max_minute=50000, min_val=10)

    acc_reached = []
    time_used = []

    for run_id in range(total_run):
        begin_time = time.time()
        acc_each_run = []
        time_each_run = []
        for time_budget_used in min_array:
            time_budget_used = time_budget_used * 60
            try:
                begin_time_u = time.time()
                U = sh.schedule_budget_per_model_based_on_T(args.search_space, time_budget_used, total_models)
                end_time_u = time.time()
                # print(f"run_id = {run_id}, time_usage for U = {end_time_u - begin_time_u}")

                begin_time_u = time.time()
                best_arch, _, B2_actual_epoch_use, _ = sh.run_phase2(U, all_models[run_id])
                end_time_u = time.time()
                # print(f"run_id = {run_id}, time_usage for run = {end_time_u - begin_time_u}")

                begin_time_u = time.time()
                acc_sh_v, _ = fgt.get_ground_truth(arch_id=best_arch, dataset=args.dataset, epoch_num=args.epoch)
                end_time_u = time.time()
                # print(f"run_id = {run_id}, get ground truth for run = {end_time_u - begin_time_u}")

                acc_each_run.append(acc_sh_v)
                time_each_run.append(time_budget_used / 60)
                print(
                    f" *********** begin with U={U}, K={len(all_models[run_id])}, "
                    f"B2_actual_epoch_use = {B2_actual_epoch_use}, acc = {acc_sh_v}, "
                    f"Given Budget = {time_budget_used} ***********")
            except Exception as e:
                acc_each_run.append(0)
                time_each_run.append(time_budget_used / 60)
        end_time = time.time()
        print(f"run_id = {run_id}, time_usage = {end_time - begin_time}")

        acc_reached.append(acc_each_run)
        time_used.append(time_each_run)

    return acc_reached, time_used


def debug_args(args, dataset):
    args.dataset = dataset
    args.device = "gpu"
    args.base_dir = "../exp_data/"
    args.is_simulate = True
    args.log_folder = "log_ku_tradeoff"

    if dataset == Config.Frappe:
        args.search_space = "mlp_sp"
        args.epoch = 13
        args.hidden_choice_len = 20

    if dataset == Config.UCIDataset:
        args.search_space = "mlp_sp"
        args.epoch = 0
        args.hidden_choice_len = 20

    if dataset == Config.Criteo:
        args.search_space = "mlp_sp"
        args.epoch = 9
        args.hidden_choice_len = 10

    if dataset == Config.c10:
        args.search_space = "nasbench201"
        args.epoch = 200

    if dataset == Config.c100:
        args.search_space = "nasbench201"
        args.epoch = 200

    if dataset == Config.imgNet:
        args.search_space = "nasbench201"
        args.epoch = 200


if __name__ == "__main__":

    args = parse_arguments()
    debug_args(args, Config.Criteo)

    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.query_api.interface import SimulateTrain
    from src.query_api.query_api_img import guess_train_one_epoch_time
    from src.eva_engine.phase2.evaluator import P2Evaluator
    from src.eva_engine.phase2.run_sh import BudgetAwareControllerSH
    from src.eva_engine.phase2.run_sr import BudgetAwareControllerSR
    from src.eva_engine.phase2.run_uniform import UniformAllocation
    from src.search_space.nas_201_api.model_params import NB201MacroCfg
    from src.search_space.nas_201_api.space import NasBench201Space
    from src.tools.io_tools import write_json
    from src.search_space.init_search_space import init_search_space

    total_run = 100
    total_models = 400

    # sample 100 * 500 models,
    space_ins = init_search_space(args)
    arch_gene = space_ins.sample_all_models()

    all_models = []
    for run_id in range(total_run):
        inner_model_list = []
        for arch_id, _ in arch_gene:
            inner_model_list.append(arch_id)
            if len(inner_model_list) == total_models:
                break
        all_models.append(inner_model_list)

    train_time_per_epoch = space_ins.profiling_train_time(dataset=args.dataset, is_simulate=True, args=args)

    fgt = SimulateTrain(space_name=args.search_space)
    evaluator = P2Evaluator(search_space_ins=space_ins,
                            dataset=args.dataset,
                            is_simulate=True)

    result_save_dic = {}

    print("--- benchmarking sh_")
    sh_ = BudgetAwareControllerSH(search_space_ins=space_ins,
                                  dataset_name=args.dataset,
                                  eta=3, time_per_epoch=train_time_per_epoch,
                                  args=args)
    acc_reached, time_used = run_one_fixed_budget_alg(sh_, train_time_per_epoch)
    result_save_dic["sh"] = {"time_used": time_used, "acc_reached": acc_reached}

    print("--- benchmarking uniform_")
    uniform_ = UniformAllocation(search_space_ins=space_ins,
                                 dataset_name=args.dataset,
                                 eta=3,
                                 time_per_epoch=train_time_per_epoch,
                                 args=args)
    acc_reached, time_used = run_one_fixed_budget_alg(uniform_, train_time_per_epoch)
    result_save_dic["uniform"] = {"time_used": time_used, "acc_reached": acc_reached}

    print("--- benchmarking sr_")
    sr_ = BudgetAwareControllerSR(search_space_ins=space_ins,
                                  dataset_name=args.dataset,
                                  eta=3,
                                  time_per_epoch=train_time_per_epoch,
                                  args=args)
    acc_reached, time_used = run_one_fixed_budget_alg(sr_, train_time_per_epoch)
    result_save_dic["sr"] = {"time_used": time_used, "acc_reached": acc_reached}

    write_json(f"{args.result_dir}/micro_phase2_{args.dataset}", result_save_dic)
