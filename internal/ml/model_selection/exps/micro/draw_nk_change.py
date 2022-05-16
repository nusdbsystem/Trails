# this is the main function of model selection.


import calendar
import os
import time
from exps.shared_args import parse_arguments
from src.common.constant import CommonVars
from multiprocessing import Process


def draw_graph2(res):
    def thousands_formatter(x, pos):
        if x >= 1e3:
            return '{:.1f}k'.format(x * 1e-3)
        else:
            return '{:.1f}'.format(x)

    print("begin")

    import json
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import matplotlib
    from matplotlib.ticker import FuncFormatter
    thousands_format = FuncFormatter(thousands_formatter)
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)

    # Helper function to load data
    def load_data(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)

    # Set your plot parameters
    bar_width = 0.25
    opacity = 0.8
    set_font_size = 15  # Set the font size
    set_lgend_size = 12
    set_tick_size = 12
    cpu_colors = ['#FFB579']
    gpu_colors = ['#3498DB']
    gpu_memory_colors = ['#98DF8A']

    hatches = ['/', '\\', 'x', 'o', 'O', '.', '*', '//', '\\\\', 'xx', 'oo', 'OO', '..', '**']
    fig, ax = plt.subplots(figsize=(6.4, 4.5))

    for index in range(len(res)):
        K, N = res[index]

        if index == len(res) - 1:
            ax.bar(index - bar_width / 2, N, bar_width, color=gpu_colors, hatch=hatches[0],
                   label='N',
                   edgecolor='black')
            ax.bar(index - bar_width / 2, K, bar_width, color=gpu_memory_colors, hatch=hatches[1],
                   bottom=N,
                   label='K',
                   edgecolor='black')
        else:
            ax.bar(index - bar_width / 2, N, bar_width, color=gpu_colors, hatch=hatches[0],
                   edgecolor='black')
            ax.bar(index - bar_width / 2, K, bar_width, color=gpu_memory_colors, hatch=hatches[1],
                   bottom=N, edgecolor='black')

        ax.bar(index + bar_width / 2, N, bar_width, color=gpu_colors, hatch=hatches[0],
               edgecolor='black')
        ax.bar(index + bar_width / 2, K, bar_width, color=gpu_memory_colors, hatch=hatches[1],
               bottom=N,
               edgecolor='black')

    ax.set_ylabel('Memory (MB)', fontsize=20)
    # ax.set_xticks(index)
    # ax.set_yscale('log')  # Set y-axis to logarithmic scale
    # ax.set_xticklabels(datasets, rotation=10, fontsize=set_font_size)
    ax.legend(fontsize=set_lgend_size)
    ax.yaxis.set_major_formatter(thousands_format)

    ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)
    # ax.set_yscale('log')
    plt.tight_layout()
    fig.tight_layout()
    # plt.show()
    print(f"saving to ./internal/ml/model_selection/exp_result/nk_bar.pdf")
    fig.savefig(f"./internal/ml/model_selection/exp_result/nk_bar.pdf",
                bbox_inches='tight')


if __name__ == "__main__":
    args = parse_arguments()

    args.is_simulate = False
    args.tfmem = CommonVars.JACFLOW
    args.models_explore = 1200
    args.search_space = "nasbench201"
    args.api_loc = "NAS-Bench-201-v1_1-096897.pth"
    args.dataset = "ImageNet16-120"
    args.batch_size = 32
    args.num_labels = 120
    args.device = "cpu"
    args.log_folder = "log_ku_tradeoff"
    args.result_dir = "./internal/ml/model_selection/exp_result"

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.eva_engine.run_ms import RunModelSelection

    rms = RunModelSelection(args.search_space, args, is_simulate=True)
    res = []
    for budget in [200, 400, 800, 1600, 3200, 6400]:
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = rms.schedule_only(
            budget=budget,
            data_loader=[None, None, None])
        print(f"K={K}, N={N}, U={U}")
        res.append((N, K))



