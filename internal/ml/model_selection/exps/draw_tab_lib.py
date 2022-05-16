from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# lines' mark size
set_marker_size = 1
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 20
set_lgend_size = 15
set_tick_size = 20

frontinsidebox = 23

# update tick size
matplotlib.rc('xtick', labelsize=set_tick_size)
matplotlib.rc('ytick', labelsize=set_tick_size)

plt.rcParams['axes.labelsize'] = set_tick_size

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]
mark_size_list = [set_marker_size, set_marker_size + 1, set_marker_size + 1, set_marker_size,
                  set_marker_size, set_marker_size, set_marker_size, set_marker_size + 1, set_marker_size + 2]
line_shape_list = ['-.', '--', '-', ':']
shade_degree = 0.2


def Add_one_line(x_time_array: list, y_twod_budget: List[List], namespace: str, index, ax):
    # training-based
    x_ = x_time_array
    y_ = y_twod_budget

    if all(isinstance(item, list) for item in x_):
        expx = np.array(x_)
        x_m = np.quantile(expx, .5, axis=0)
    else:
        x_m = x_

    exp = np.array(y_)
    exp = np.where(exp > 10, exp, exp * 100)

    y_h = np.quantile(exp, .75, axis=0)
    y_m = np.quantile(exp, .5, axis=0)
    y_l = np.quantile(exp, .25, axis=0)

    ax.plot(x_m, y_m,
            line_shape_list[int(index % len(line_shape_list))],
            label=namespace,
            linewidth=8,
            )

    ax.fill_between(x_m, y_l, y_h, alpha=shade_degree)
    return x_m.tolist(), y_m.tolist()


def draw_structure_data_anytime(
        all_lines: List,
        dataset: str, name_img: str, max_value,
        figure_size=(6.8, 4.8),
        annotations=[],
        x_ticks=None, y_ticks=None, unique_labels=None,
        x_label_name=r"Response Time Threshold $T_{max}$ (min)"
):
    fig, ax = plt.subplots(figsize=figure_size)

    # draw all lines
    time_usage = []
    y_achieved = []
    for i, each_line_info in enumerate(all_lines):
        _x_array = each_line_info[0]
        _y_2d_array = each_line_info[1]
        _name_space = each_line_info[2]
        time_arr, y_array = Add_one_line(_x_array, _y_2d_array, _name_space, i, ax, )
        time_usage.append(time_arr)
        y_achieved.append(y_array)

    try:
        # find the time to reach a target AUC
        RENAS_Time = 0.00000000001
        if max(y_achieved[0]) < max_value * 0.01:
            print(f"EANAS cannot reach {max_value * 0.01} it max value is {max(y_achieved[0])}")
        for i in range(len(y_achieved[0])):
            if y_achieved[0][i] >= max_value * 0.01:
                RENAS_Time = time_usage[0][i]
        if RENAS_Time == 0:
            print("RENAS_Time Miss it")

        ATAS_Time = 0.00000000001
        if max(y_achieved[1]) < max_value:
            print(f"ATLAS cannot reach {max_value} it max value is {max(y_achieved[1])}")
        for i in range(len(y_achieved[1])):
            if y_achieved[1][i] >= max_value:
                ATAS_Time = time_usage[1][i]
        if ATAS_Time == 0:
            print("ATAS_Time Miss it")
        print(f"speed-up on {dataset} = {RENAS_Time / ATAS_Time}, "
              f"t_train = {RENAS_Time}, t_f = {ATAS_Time}")
    except:
        pass

    # plt.xscale("log")
    # plt.grid()
    # plt.xlabel(r"Time Budget $T$ (min)", fontsize=set_font_size)
    # plt.ylabel(f"AUC on {dataset.upper()}", fontsize=set_font_size)

    plt.xscale("log")
    ax.grid()
    ax.set_xlabel(x_label_name, fontsize=set_font_size)
    ax.set_ylabel(f"AUC on {dataset.upper()}", fontsize=set_font_size)
    # ax.set_xscale("log")
    # ax.set_xlim(0.001, 10e4)
    # ax.set_ylim(x1_lim[0], x1_lim[1])

    if y_ticks is not None:
        if y_ticks[0] is not None:
            ax.set_ylim(bottom=y_ticks[0])
        if y_ticks[1] is not None:
            ax.set_ylim(top=y_ticks[1])
        # ax.set_ylim(y_ticks[0], y_ticks[1])
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels(y_ticks)
    if x_ticks is not None:
        if x_ticks[0] is not None:
            ax.set_xlim(left=x_ticks[0])
        if x_ticks[1] is not None:
            ax.set_xlim(right=x_ticks[1])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))

    if max_value > 0:
        plt.axhline(max_value, color='r', linestyle='-', label='Global Best AUC', linewidth=3)

    for i in range(len(annotations)):
        ele = annotations[i]
        ax.plot(ele[2], ele[1], mark_list[i], label=ele[0], markersize=set_marker_point)

    # export_legend(fig, filename="any_time_legend", unique_labels=['Global Best AUC', "TabNAS", "RE-NAS",  "ATLAS", ])
    export_legend(fig, filename="phase2_micro_legend", unique_labels=["UNIFORM", "SUCCREJCT", "SUCCHALF"])
    # export_legend(ori_fig=fig, colnum=5, unique_labels=unique_labels)
    plt.tight_layout()

    print(f"saving to {name_img}.pdf")
    fig.savefig(f"{name_img}.pdf", bbox_inches='tight')


def draw_structure_data_anytime_system_version(
        all_lines: List,
        dataset: str, name_img: str, max_value,
        figure_size=(6.4, 4.5),
        annotations=[],
        x_ticks=None, y_ticks=None, unique_labels=None,
        x_label_name=r"Response Time Threshold $T_{max}$ (min)"
):
    global set_marker_size, mark_size_list
    set_marker_size = 15
    mark_size_list = [set_marker_size, set_marker_size + 1, set_marker_size + 1, set_marker_size,
                      set_marker_size, set_marker_size, set_marker_size, set_marker_size + 1, set_marker_size + 2]

    from matplotlib.ticker import FuncFormatter

    def thousands_formatter(x, pos):
        return '{:.2f}'.format(x)

    thousands_format = FuncFormatter(thousands_formatter)

    fig, ax = plt.subplots(figsize=figure_size)

    info_train = all_lines[0]
    info_train_free = all_lines[1]
    info_two_phase = all_lines[2]

    if all(isinstance(item, list) for item in info_train[0]):
        expx = np.array(info_train[0])
        x_m = np.quantile(expx, .5, axis=0)
    else:
        x_m = info_train[0]
    exp = np.array(info_train[1]) * 100
    sys_acc_p1_h = np.quantile(exp, .75, axis=0)
    sys_acc_p1_m = np.quantile(exp, .5, axis=0)
    sys_acc_p1_l = np.quantile(exp, .25, axis=0)
    ax.plot(x_m, sys_acc_p1_m, mark_list[-3] + line_shape_list[0], label="Training-Based MS",
            markersize=mark_size_list[-3])
    ax.fill_between(x_m, sys_acc_p1_l, sys_acc_p1_h, alpha=shade_degree)

    # plot simulate result of system
    if all(isinstance(item, list) for item in info_train_free[0]):
        expx = np.array(info_train_free[0])
        x_m = np.quantile(expx, .5, axis=0)
    else:
        x_m = info_train_free[0]
    exp = np.array(info_train_free[1]) * 100
    sys_acc_p1_h = np.quantile(exp, .75, axis=0)
    sys_acc_p1_m = np.quantile(exp, .5, axis=0)
    sys_acc_p1_l = np.quantile(exp, .25, axis=0)
    ax.plot(x_m, sys_acc_p1_m, mark_list[-2] + line_shape_list[1], label="Training-Free MS",
            markersize=mark_size_list[-2])
    ax.fill_between(x_m, sys_acc_p1_l, sys_acc_p1_h, alpha=shade_degree)

    # plot simulate result of system
    if all(isinstance(item, list) for item in info_two_phase[0]):
        expx = np.array(info_two_phase[0])
        x_m = np.quantile(expx, .5, axis=0)
    else:
        x_m = info_two_phase[0]
    exp = np.array(info_two_phase[1]) * 100
    sys_acc_h = np.quantile(exp, .75, axis=0)
    sys_acc_m = np.quantile(exp, .5, axis=0)
    sys_acc_l = np.quantile(exp, .25, axis=0)
    ax.plot(x_m, sys_acc_m, mark_list[-1] + line_shape_list[2], label="2Phase-MS", markersize=mark_size_list[-1])
    ax.fill_between(x_m, sys_acc_l, sys_acc_h, alpha=shade_degree)

    for i in range(len(annotations)):
        ele = annotations[i]
        # this is to make the second plot the right color.
        ax.plot(ele[2], ele[1] * 0.00001, "s", label=ele[0],
                markersize=set_marker_point)
        ax.plot(ele[2] / 60, ele[1], "s", label=ele[0],
                markersize=set_marker_point)

    plt.xscale("log")
    ax.grid()
    plt.xlabel(r"Response Time Threshold $T_{max}$ (min)", fontsize=set_font_size)
    ax.set_ylabel(f"Test AUC on {dataset.upper()}", fontsize=set_font_size)

    ax.xaxis.label.set_size(set_tick_size)
    ax.yaxis.label.set_size(set_tick_size)

    # ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    ax.axhline(max_value, color='r', linestyle='-', label='Global Best Accuracy')

    tick_values = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f'$10^{{{int(np.log10(val))}}}$' for val in tick_values])

    if y_ticks is not None:
        if y_ticks[0] is not None:
            ax.set_ylim(bottom=y_ticks[0])
        if y_ticks[1] is not None:
            ax.set_ylim(top=y_ticks[1])
        # ax.set_ylim(y_ticks[0], y_ticks[1])
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels(y_ticks)
    if x_ticks is not None:
        if x_ticks[0] is not None:
            ax.set_xlim(left=x_ticks[0])
        if x_ticks[1] is not None:
            ax.set_xlim(right=x_ticks[1])

    ax.yaxis.set_major_formatter(thousands_format)
    # this is for unique hash
    # export_legend(
    #     fig,
    #     colnum=3,
    #     unique_labels=['TE-NAS (Training-Free)', 'ENAS (Weight sharing)',
    #                    'KNAS (Training-Free)', 'DARTS-V1 (Weight sharing)', 'DARTS-V2 (Weight sharing)',
    #                    'Training-Based MS', 'Training-Free MS', '2Phase-MS', 'Global Best Accuracy'])
    plt.tight_layout()
    print(f"saving to {name_img}.pdf")
    fig.savefig(f"{name_img}.pdf", bbox_inches='tight')


def draw_structure_data_anytime_system_version_imageNetFULL(
        all_lines: List,
        dataset: str, name_img: str, max_value,
        figure_size=(6.4, 4.5),
        annotations=[],
        x_ticks=None, y_ticks=None, unique_labels=None,
        x_label_name=r"Response Time Threshold $T_{max}$ (min)"
):
    global set_marker_size, mark_size_list
    set_marker_size = 15
    mark_size_list = [set_marker_size, set_marker_size + 1, set_marker_size + 1, set_marker_size,
                      set_marker_size, set_marker_size, set_marker_size, set_marker_size + 1, set_marker_size + 2]

    from matplotlib.ticker import FuncFormatter

    def thousands_formatter(x, pos):
        return '{:.2f}'.format(x)

    thousands_format = FuncFormatter(thousands_formatter)

    fig, ax = plt.subplots(figsize=figure_size)

    info_train = all_lines[0]
    info_two_phase = all_lines[1]

    # plot simulate result of system
    # plot simulate result of system
    if all(isinstance(item, list) for item in info_two_phase[0]):
        expx = np.array(info_two_phase[0])
        x_m = np.quantile(expx, .5, axis=0)
    else:
        x_m = info_two_phase[0]
    exp = np.array(info_two_phase[1]) * 78.4 / 47
    sys_acc_h = np.quantile(exp, .75, axis=0)
    sys_acc_m = np.quantile(exp, .5, axis=0)
    sys_acc_l = np.quantile(exp, .25, axis=0)
    ax.plot(x_m, sys_acc_m, mark_list[-1] + line_shape_list[2], label="2Phase-MS", markersize=mark_size_list[-1])
    ax.fill_between(x_m, sys_acc_l, sys_acc_h, alpha=shade_degree)

    print(x_m, sys_acc_m)

    for i in range(len(annotations)):
        ele = annotations[i]
        # this is to make the second plot the right color.
        # ax.plot(ele[2], ele[1] * 0.00001, "s", label=ele[0],
        #         markersize=set_marker_point)
        if i == 1:
            ax.plot(ele[2], ele[1], mark_list[i], label=ele[0],
                    markersize=set_marker_point + 2.5)
        else:
            ax.plot(ele[2], ele[1], mark_list[i], label=ele[0],
                    markersize=set_marker_point)

    plt.xscale("log")
    ax.grid()
    plt.xlabel(r"Response Time Threshold $T_{max}$ (min)", fontsize=set_font_size)
    ax.set_ylabel(f"Top-1 Acc on ImageNet", fontsize=set_font_size)

    ax.xaxis.label.set_size(set_tick_size)
    ax.yaxis.label.set_size(set_tick_size)

    # ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

    ax.axhline(max_value * 78.4 / 47, color='r', linestyle='-', label='Global Best Accuracy')
    # ax.axvline(207713.4121265026/60, color='b', linestyle='-')

    # tick_values = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    # ax.set_xticks(tick_values)
    # ax.set_xticklabels([f'$10^{{{int(np.log10(val))}}}$' for val in tick_values])

    if y_ticks is not None:
        if y_ticks[0] is not None:
            ax.set_ylim(bottom=y_ticks[0] * 78.4 / 47)
        if y_ticks[1] is not None:
            ax.set_ylim(top=y_ticks[1] * 78.4 / 47)
        # ax.set_ylim(y_ticks[0], y_ticks[1])
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels(y_ticks)
    if x_ticks is not None:
        if x_ticks[0] is not None:
            ax.set_xlim(left=x_ticks[0])
        if x_ticks[1] is not None:
            ax.set_xlim(right=x_ticks[1])

    ax.legend(fontsize=set_lgend_size, loc='lower right')
    ax.yaxis.set_major_formatter(thousands_format)
    # this is for unique hash
    # export_legend(
    #     fig,
    #     colnum=3,
    #     unique_labels=['TE-NAS (Training-Free)', 'ENAS (Weight sharing)',
    #                    'KNAS (Training-Free)', 'DARTS-V1 (Weight sharing)', 'DARTS-V2 (Weight sharing)',
    #                    'Training-Based MS', 'Training-Free MS', '2Phase-MS', 'Global Best Accuracy'])
    plt.tight_layout()
    print(f"saving image to {name_img}.pdf")
    fig.savefig(f"{name_img}.pdf", bbox_inches='tight')


def export_legend(ori_fig, filename="any_time_legend", colnum=9, unique_labels=None):
    if unique_labels is None:
        unique_labels = []
    fig2 = plt.figure(figsize=(5, 0.3))
    lines_labels = [ax.get_legend_handles_labels() for ax in ori_fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # grab unique labels
    if len(unique_labels) == 0:
        unique_labels = set(labels)
    # assign labels and legends in dict
    legend_dict = dict(zip(labels, lines))
    # query dict based on unique labels
    unique_lines = [legend_dict[x] for x in unique_labels]
    fig2.legend(unique_lines, unique_labels, loc='center',
                ncol=colnum,
                fancybox=True,
                shadow=True, scatterpoints=1, fontsize=15)
    fig2.tight_layout()
    fig2.savefig(f"{filename}.pdf", bbox_inches='tight')


import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(data: List, fontsize: int,
                 x_array_name: str, y_array_name: str,
                 title: str, output_file: str,
                 decimal_places: int,
                 u_ticks, k_ticks,
                 ):
    labelsize = fontsize
    # Convert the data to a NumPy array
    data_array = np.array(data)

    # Custom annotation function
    def custom_annot(val):
        return "{:.{}f}".format(val, decimal_places) if val > 0 else ""

    # Convert the custom annotations to a 2D array
    annot_array = np.vectorize(custom_annot)(data_array)

    # Create a masked array to hide the cells with values less than or equal to 0
    masked_data = np.ma.masked_array(data_array, data_array <= 0)

    # Set the figure size (width, height) in inches
    fig, ax = plt.subplots(figsize=(8, 4))

    # Use the "viridis" colormap
    cmap = "viridis"

    # Create a heatmap
    sns.heatmap(masked_data, annot=annot_array, fmt='', cmap=cmap, mask=masked_data.mask, ax=ax,
                annot_kws={"size": fontsize, "ha": "center", "va": "center"},
                xticklabels=u_ticks, yticklabels=k_ticks)

    # Set axis labels
    ax.set_xlabel(x_array_name, fontsize=fontsize)
    ax.set_ylabel(y_array_name, fontsize=fontsize)

    # Set x/y-axis tick size
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    # Set the title
    # ax.set_title(title, fontsize=fontsize)

    # Set tight layout
    plt.tight_layout()

    # Save the plot to a PDF file
    plt.savefig(output_file)
