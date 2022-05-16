# this is the main function of model selection.
import copy

import numpy as np
import calendar
import os
import time
from exps.shared_args import parse_arguments
from multiprocessing import Process


def run_with_time_budget(time_budget: float, only_phase1: bool):
    """
    :param time_budget: the given time budget, in second
    :return:
    """

    best_arch, best_arch_performance, time_usage, _, _, _, p1_trace_highest_score, p1_trace_highest_scored_models_id = \
        rms.select_model_online(
            budget=time_budget,
            data_loader=[None, None, None],
            only_phase1=only_phase1,
            run_workers=1)

    return best_arch, best_arch_performance, time_usage, \
           p1_trace_highest_score, p1_trace_highest_scored_models_id


def draw_graph(result_m, kn_rate_list_m, dataset, kn_rate_list_l, kn_rate_list_h):
    """
    kn_rate_list_m: x array indexs
    result_m: y array indexs for each line
    """
    import matplotlib.pyplot as plt
    import matplotlib
    from exps.draw_tab_lib import export_legend
    set_line_width = 5
    set_tick_size = 25
    # update tick size
    matplotlib.rc('xtick', labelsize=set_tick_size)
    matplotlib.rc('ytick', labelsize=set_tick_size)
    plt.rcParams['axes.labelsize'] = set_tick_size
    mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]
    line_shape_list = ['-.', '--', '-', ':']

    # this is for the same size in figure
    if dataset == "DIABETES":
        fig2, ax = plt.subplots(figsize=(7, 4))
    elif dataset == "FRAPPE":
        fig2, ax = plt.subplots(figsize=(7, 4.15))
    else:
        fig2, ax = plt.subplots(figsize=(7, 4.2))

    # this is to plot trade off between N and K
    unique_labels = []
    for i, (time_budget_key, y_array) in enumerate(result_m.items()):
        ax.plot(kn_rate_list_m, y_array,
                mark_list[i % len(mark_list)] + line_shape_list[i % len(line_shape_list)],
                label=r"$T$=" + time_budget_key,
                linewidth=set_line_width,
                markersize=5
                )

        # plt.fill_between(y_array, kn_rate_list_l[time_budget_key], kn_rate_list_h[time_budget_key], alpha=shade_degree)

        unique_labels.append(r"$T$=" + time_budget_key)
    ax.set_xscale("log")
    ax.grid()
    ax.set_xlabel("M/K")
    ax.set_ylabel(f"AUC on {dataset}")
    # plt.ylim(y_lim[0], y_lim[1])
    # export_legend(fig2, "trade_off_nk_legend", unique_labels=unique_labels)
    plt.legend(ncol=2, prop={'size': 14})
    # plt.show()
    fig2.savefig(f"./internal/ml/model_selection/exp_result/trade_off_nk_{dataset}.pdf", bbox_inches='tight')


def convert_to_two_dim_list(original_list, len_k, len_u):
    """
    Device the original_list into len_k sub-list, each with len_u elements
    Return a two dimension list
    """
    if len_k * len_u > len(original_list):
        print("Error: len_k * len_u > len(original_list). Cannot proceed.")
        return
    two_dim_list = [original_list[i * len_u: (i + 1) * len_u] for i in range(len_k)]
    return two_dim_list


def debug_args(args, dataset):
    if dataset == Config.c10:
        args.tfmem = "jacflow"
        args.search_space = "nasbench201"
        args.api_loc = "NAS-Bench-201-v1_1-096897.pth"
        args.epoch = 200
        args.dataset = "cifar10"
        args.base_dir = "../exp_data/"
        args.is_simulate = True
        args.log_folder = "log_ku_tradeoff"


    if dataset == Config.c100:
        args.tfmem = "jacflow"
        args.search_space = "nasbench201"
        args.api_loc = "NAS-Bench-201-v1_1-096897.pth"
        args.epoch = 200
        args.dataset = "cifar100"
        args.base_dir = "../exp_data/"
        args.is_simulate = True
        args.log_folder = "log_ku_tradeoff"

    if dataset == Config.imgNet:
        args.tfmem = "jacflow"
        args.search_space = "nasbench201"
        args.api_loc = "NAS-Bench-201-v1_1-096897.pth"
        args.epoch = 200
        args.dataset = "ImageNet16-120"
        args.base_dir = "../exp_data/"
        args.is_simulate = True
        args.log_folder = "log_ku_tradeoff"

    # tabular
    if dataset == Config.Frappe:
        args.tfmem = "jacflow"
        args.search_space = "mlp_sp"
        args.epoch = 14
        args.dataset = "frappe"
        args.base_dir = "../exp_data/"
        args.is_simulate = True
        args.log_folder = "log_ku_tradeoff"
        args.hidden_choice_len = 20

    if dataset == Config.Criteo:
        args.tfmem = "jacflow"
        args.search_space = "mlp_sp"
        args.epoch = 10
        args.dataset = "criteo"
        args.base_dir = "../exp_data/"
        args.is_simulate = True
        args.log_folder = "log_ku_tradeoff"
        args.hidden_choice_len = 10


if __name__ == "__main__":
    args = parse_arguments()
    from src.common.constant import Config

    # dataset = Config.Criteo
    # dataset = Config.Frappe
    # dataset = Config.c10
    # dataset = Config.c100
    dataset = Config.imgNet

    debug_args(args, dataset)
    default_kn_rate_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.eva_engine.run_ms import RunModelSelection
    from src.logger import logger
    from src.tools.io_tools import write_json, read_json
    import json

    # for this exp, we repeat 100 times and set max to 1000 mins

    rms = RunModelSelection(args.search_space, args, is_simulate=True)

    # Fix budget to 100 mins and only use phase1, try differetn K and U
    while True:
        # input = 3|[4, 8, 16, 32]|[1, 2, 4, 8, 16, 32, 64, 128, 256]
        items = input()
        budget_array = json.loads(items.split("|")[1])
        total_run = int(items.split("|")[0])
        kn_rate_list = json.loads(items.split("|")[2])

        if kn_rate_list != default_kn_rate_list:
            result = {}
            default_kn_rate_list = kn_rate_list
        else:
            result = read_json(f'{args.result_dir}/nk_{args.dataset}_{args.tfmem}.json')

        #
        # if args.dataset in [Config.c10, Config.c100, Config.imgNet]:
        #     budget_array = [1, 2, 4, 8, 16, 32, 64, 128]
        # else:
        #     if args.dataset == Config.Criteo:
        #         budget_array = [16, 64, 256, 1024]
        #     elif args.dataset == Config.Frappe:
        #         budget_array = [4, 8, 16, 32]
        #     elif args.dataset == Config.UCIDataset:
        #         budget_array = [0.1, 0.5, 1, 2]

        # result of time_budget: value for each kn rate
        print(f"run with {budget_array}, and {total_run}, {kn_rate_list}")
        if len(result) != -1:
            # result = {}
            for time_budget in budget_array:
                if str(time_budget) + " min" not in result:
                    result[str(time_budget) + " min"] = []
                time_budget_sec = time_budget * 60
                # For each time budget, repeat multiple times.
                for run_id in range(total_run):
                    run_list = []
                    for kn_rate in kn_rate_list:
                        args.kn_rate = kn_rate
                        logger.info(f"\n Running job with budget={time_budget} min \n")

                        try:
                            best_arch, best_arch_performance, time_usage, \
                            p1_trace_highest_score, p1_trace_highest_scored_models_id = \
                                run_with_time_budget(time_budget_sec, only_phase1=False)
                            run_list.append(best_arch_performance)
                        except Exception as e:
                            print(f"Errored at the time_sec = {time_budget_sec}, error = {e}")
                            # cannot schedule!
                            run_list.append(0)

                    result[str(time_budget) + " min"].append(run_list)
            print("Done")
            write_json(f'{args.result_dir}/nk_{args.dataset}_{args.tfmem}.json', result)

        # put your scaled_data and two_d_epoch here
        if args.dataset == Config.Criteo:
            dataset_name = "CRITEO"
        if args.dataset == Config.UCIDataset:
            dataset_name = "DIABETES"
        if args.dataset == Config.Frappe:
            dataset_name = "FRAPPE"

        if args.dataset == Config.c10:
            dataset_name = "C10"
        if args.dataset == Config.c100:
            dataset_name = "C100"
        if args.dataset == Config.imgNet:
            dataset_name = "IN-16"

        # Record the medium value
        result2 = copy.deepcopy(result)
        result_lower = {}
        result_upper = {}
        for time_budget in budget_array:
            lst = np.quantile(np.array(result2[str(time_budget) + " min"]), .5, axis=0).tolist()
            result[str(time_budget) + " min"] = [ele * 100 for ele in lst]
            lst_lower = np.quantile(np.array(result2[str(time_budget) + " min"]), .25, axis=0).tolist()
            result_lower[str(time_budget) + " min"] = [ele * 100 for ele in lst_lower]
            lst_upper = np.quantile(np.array(result2[str(time_budget) + " min"]), .75, axis=0).tolist()
            result_upper[str(time_budget) + " min"] = [ele * 100 for ele in lst_upper]

        p = Process(target=draw_graph, args=(result, kn_rate_list, dataset_name, result_lower, result_upper))
        p.start()
        p.join()
