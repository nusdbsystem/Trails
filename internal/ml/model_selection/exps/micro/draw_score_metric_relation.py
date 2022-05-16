from src.tools.compute import sample_in_log_scale_new
from exps.draw_tab_lib import export_legend
from src.tools.io_tools import read_json
from typing import List

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
# import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

# lines' mark size
set_marker_size = 1
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 20
set_lgend_size = 15
set_tick_size = 15

frontinsidebox = 23

# update tick size
matplotlib.rc('xtick', labelsize=set_tick_size)
matplotlib.rc('ytick', labelsize=set_tick_size)

plt.rcParams['axes.labelsize'] = set_tick_size

mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]
mark_size_list = [set_marker_size, set_marker_size + 1, set_marker_size + 1, set_marker_size,
                  set_marker_size, set_marker_size, set_marker_size, set_marker_size + 1, set_marker_size + 2]
line_shape_list = ['-.', '--', '-', ':']
color_list = ['blue', 'green', 'black', 'purple', 'brown', 'red']
shade_degree = 0.2


def Add_one_line(x_time_array: list, y_twod_budget: List[List], namespace: str, index, ax):
    if index == 1:
        # Create a new y-axis that shares the same x-axis
        ax = ax.twinx()
        ax.set_ylabel(f"ExpressFlow score", fontsize=set_font_size)

    # training-based
    x_ = x_time_array
    y_ = y_twod_budget

    if all(isinstance(item, list) for item in x_):
        expx = np.array(x_)
        x_m = np.quantile(expx, .5, axis=0)
    else:
        x_m = x_

    exp = np.array(y_) * 100
    y_h = np.quantile(exp, .75, axis=0)
    y_m = np.quantile(exp, .5, axis=0)
    y_l = np.quantile(exp, .25, axis=0)

    ax.plot(x_m, y_m,
            mark_list[index - 3] + line_shape_list[index],
            color=color_list[index],
            label=namespace,
            markersize=mark_size_list[index - 3],
            linewidth=5
            )
    ax.fill_between(x_m, y_l, y_h, alpha=shade_degree)
    return x_m


def draw_structure_data_anytime(
        all_lines: List,
        dataset: str, name_img: str, max_value,
        figure_size=(6.4, 4.5),
        annotations=[],
        x_ticks=None, y_ticks=None, unique_labels=None):
    fig, ax = plt.subplots(figsize=figure_size)

    # draw all lines
    time_usage = []
    for i, each_line_info in enumerate(all_lines):
        _x_array = each_line_info[0]
        _y_2d_array = each_line_info[1]
        _name_space = each_line_info[2]
        time_arr = Add_one_line(_x_array, _y_2d_array, _name_space, i, ax)
        time_usage.append(time_arr)

    plt.xscale("log")
    ax.grid()
    ax.set_xlabel(r"Number of explored architectures", fontsize=set_font_size)
    ax.set_ylabel(f"AUC on {dataset.upper()}", fontsize=set_font_size)

    if y_ticks is not None:
        if y_ticks[0] is not None:
            ax.set_ylim(bottom=y_ticks[0])
        if y_ticks[1] is not None:
            ax.set_ylim(top=y_ticks[1])
    if x_ticks is not None:
        if x_ticks[0] is not None:
            ax.set_xlim(left=x_ticks[0])
        if x_ticks[1] is not None:
            ax.set_xlim(right=x_ticks[1])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))

    if max_value > 0:
        plt.axhline(max_value, color='r', linestyle='-', label='Global Best AUC')

    for i in range(len(annotations)):
        ele = annotations[i]
        ax.plot(ele[2], ele[1], mark_list[i], label=ele[0], markersize=set_marker_point)

    export_legend(ori_fig=fig, colnum=5, unique_labels=unique_labels)
    plt.tight_layout()

    fig.savefig(f"{name_img}.pdf", bbox_inches='tight')


def sample_some_points(x_array, y_2d_array, save_points, remove_n_points=1) -> (List, List):
    result_x_array = []
    result_y_array = []
    for run_id, time_list in enumerate(x_array):
        indices = sample_in_log_scale_new(time_list, save_points)
        # Sample the list using the calculated indices
        each_run_x_array = [time_list[i] for i in indices]
        each_run_y_array = [y_2d_array[run_id][int(i)] for i in indices]

        if remove_n_points != 0:
            result_x_array.append(each_run_x_array[:-remove_n_points])
            result_y_array.append(each_run_y_array[:-remove_n_points])
        else:
            result_x_array.append(each_run_x_array)
            result_y_array.append(each_run_y_array)

    return result_x_array, result_y_array


# 'frappe, criteo, uci_diabetes'
# dataset = "frappe"
# dataset = "criteo"
dataset = "uci_diabetes"
result_dir = "./internal/ml/model_selection/exp_result/"
kn_rate = -1
num_points = 12

if dataset == "uci_diabetes":
    epoch = 0
    mx_value = 67.4
    y_lim = [None, None]
    x_ticks = [1.2, 1600]
    figure_size = (6.4, 4)
    datasetfg_name = "Diabetes"
    remove_n_points = 0
    annotations = []
    ea_auc_file = ""
    ea_score_file = ""
elif dataset == "frappe":
    epoch = 13
    mx_value = 98.08
    y_lim = [None, None]
    x_ticks = [1.2, 1600]
    figure_size = (6.4, 4)
    datasetfg_name = dataset
    remove_n_points = 0
    annotations = []
    ea_auc_file = ""
    ea_score_file = ""

elif dataset == "criteo":
    epoch = 9
    mx_value = 80.335
    y_lim = [None, None]
    x_ticks = [1.2, 1600]
    figure_size = (6.4, 4)
    datasetfg_name = dataset
    annotations = []
    remove_n_points = 0
else:
    pass

checkpoint_name = read_json(f"{result_dir}/re_{dataset}_{kn_rate}_{num_points}_auc.json")
checkpoint_score_name = read_json(f"{result_dir}/re_{dataset}_{kn_rate}_{num_points}_score.json")

sampled_auc_x, sampled_auc_y = sample_some_points(
    x_array=checkpoint_name["explored_arch"],
    y_2d_array=checkpoint_name["achieved_value"],
    save_points=9,
    remove_n_points=remove_n_points)

sampled_score_x, sampled_score_y = sample_some_points(
    x_array=checkpoint_score_name["explored_arch"],
    y_2d_array=checkpoint_score_name["achieved_value"],
    save_points=9,
    remove_n_points=remove_n_points)

all_lines = [
    [sampled_auc_x, sampled_auc_y, "AUC of searched architecture"],
    [sampled_score_x, sampled_score_y, "ExpressFlow score"],
]

print(f"saving to ./internal/ml/model_selection/exp_result/p1_score_auc_{dataset}")
draw_structure_data_anytime(
    all_lines=all_lines,
    dataset=datasetfg_name,
    name_img=f"./internal/ml/model_selection/exp_result/p1_score_auc_{dataset}",
    max_value=-1,
    figure_size=figure_size,
    annotations=annotations,
    y_ticks=y_lim,
    x_ticks=x_ticks,
    unique_labels=["AUC of searched architecture", "ExpressFlow score"]
)
