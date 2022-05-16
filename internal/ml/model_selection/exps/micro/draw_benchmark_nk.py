# this is the main function of model selection.
import copy

import numpy as np
import calendar
import os
import time
from exps.shared_args import parse_arguments
from multiprocessing import Process
from matplotlib.ticker import FuncFormatter


def draw_graph(result_m, kn_rate_list_m, dataset, kn_rate_list_l, kn_rate_list_h):
    """
    kn_rate_list_m: x array indexs
    result_m: y array indexs for each line
    """
    import matplotlib.pyplot as plt

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

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
        fig2, ax = plt.subplots(figsize=(6.8, 4.8))
    elif dataset == "FRAPPE":
        fig2, ax = plt.subplots(figsize=(6.8, 4.8))
    else:
        fig2, ax = plt.subplots(figsize=(6.8, 4.8))

    # this is to plot trade off between N and K
    unique_labels = []
    for i, (time_budget_key, y_array) in enumerate(result_m.items()):
        ax.plot(kn_rate_list_m, y_array,
                mark_list[i % len(mark_list)] + line_shape_list[i % len(line_shape_list)],
                label=r"$T_{max}$=" + time_budget_key,
                linewidth=8,
                markersize=0
                )

        # plt.fill_between(y_array, kn_rate_list_l[time_budget_key], kn_rate_list_h[time_budget_key], alpha=shade_degree)

        unique_labels.append(r"$T_{max}$=" + time_budget_key)

    def thousands_formatter(x, pos):
        return '{:.2f}'.format(x)

    formatter = FuncFormatter(thousands_formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xscale("log")
    ax.grid()
    ax.set_xlabel("N/K")
    ax.set_ylabel(f"Test AUC on {dataset}")
    # plt.ylim(y_lim[0], y_lim[1])
    export_legend(fig2, "trade_off_nk_legend", unique_labels=unique_labels)
    # plt.legend(ncol=2, prop={'size': 14})
    # plt.show()
    print(f"saving to ./internal/ml/model_selection/exp_result/trade_off_nk_{dataset}.pdf", )
    fig2.savefig(f"./internal/ml/model_selection/exp_result/trade_off_nk_{dataset}.pdf", bbox_inches='tight')


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

    if dataset == Config.Criteo:
        args.tfmem = "jacflow"
        args.search_space = "mlp_sp"
        args.epoch = 10
        args.dataset = "criteo"
        args.base_dir = "../exp_data/"
        args.is_simulate = True
        args.log_folder = "log_ku_tradeoff"


from src.tools.io_tools import write_json, read_json
from src.common.constant import Config

# dataset = Config.Criteo
dataset = Config.Frappe
# dataset = Config.c10
# dataset = Config.c100
# dataset = Config.imgNet

args = parse_arguments()
debug_args(args, dataset)

budget_array = [4, 8, 16, 32]
kn_rate_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
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

result = read_json(f'{args.result_dir}/nk_{args.dataset}_{args.tfmem}.json')
print(f"read from {args.result_dir}/nk_{args.dataset}_{args.tfmem}.json")
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

draw_graph(result, kn_rate_list, dataset_name, result_lower, result_upper)
