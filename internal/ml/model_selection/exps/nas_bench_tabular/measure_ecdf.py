
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from src.tools.io_tools import read_json

# lines' mark size
set_marker_size = 15
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 25
set_lgend_size = 15
set_tick_size = 20

frontinsidebox = 23

# update tick size
matplotlib.rc('xtick', labelsize=set_tick_size)
matplotlib.rc('ytick', labelsize=set_tick_size)

plt.rcParams['axes.labelsize'] = set_tick_size

mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]
mark_size_list = [set_marker_size, set_marker_size + 1, set_marker_size + 1, set_marker_size,
                  set_marker_size, set_marker_size, set_marker_size, set_marker_size + 1, set_marker_size + 2]
line_shape_list = ['-.', '--', '-', ':']
shade_degree = 0.2
base_dir = "../exp_data/"


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
                shadow=True, scatterpoints=1, fontsize=set_lgend_size)
    fig2.tight_layout()
    fig2.savefig(f"{filename}.pdf", bbox_inches='tight')


def draw_edcf():
    # extract train_auc and valid_auc into separate lists
    for dataset, architectures in data_dict.items():

        fig, ax = plt.subplots(figsize=(6.4, 3.5))
        print(dataset)
        train_auc = []
        valid_auc = []
        for architecture, epochs in architectures.items():
            for epoch, metrics in epochs.items():
                if str(epoch_sampled[dataset]) == epoch:
                    train_auc.append(metrics["train_auc"])
                    valid_auc.append(metrics["valid_auc"])
                    break

        print("Train Standard Deviation:", np.std(train_auc), "mean = ", np.mean(train_auc), "max =", max(train_auc), "min =", min(train_auc))
        print("Valid Standard Deviation:", np.std(valid_auc), "mean = ", np.mean(valid_auc), "max =", max(valid_auc), "min =", min(valid_auc))

        # calculate and plot ECDF for train_auc
        sorted_train_auc = np.sort(train_auc)
        y_train = np.arange(1, len(sorted_train_auc) + 1) / len(sorted_train_auc)
        plt.plot(sorted_train_auc, y_train, label='Training  AUC', linewidth=3, linestyle='--')

        # calculate and plot ECDF for valid_auc
        sorted_valid_auc = np.sort(valid_auc)
        y_valid = np.arange(1, len(sorted_valid_auc) + 1) / len(sorted_valid_auc)
        plt.plot(sorted_valid_auc, y_valid, label='Validation AUC', linewidth=3, linestyle='-')

        y_m = np.quantile(sorted_valid_auc, .5, axis=0)
        print("epoch", epoch_sampled[dataset], "medium", y_m, "best", max(sorted_valid_auc))
        # plt.xlim(left=0.45)

        plt.grid()
        plt.xlabel('Accuracy')
        plt.ylabel('ECDF')
        # plt.legend(loc='upper left', fontsize=set_lgend_size)
        plt.tight_layout()
        export_legend(ori_fig=fig, colnum=5)
        print(f"saving to  space_{dataset}_{epoch_sampled[dataset]}.pdf")
        fig.savefig(f"space_{dataset}_{epoch_sampled[dataset]}.pdf", bbox_inches='tight')


dataset_used = "frappe"
# dataset_used = "uci_diabetes"
# dataset_used = "criteo"


if dataset_used == "frappe":
    mlp_train_frappe = os.path.join(
        base_dir,
        "tab_data/frappe/all_train_baseline_frappe.json")
    data_dict = read_json(mlp_train_frappe)
elif dataset_used == "uci_diabetes":
    mlp_train_uci_diabetes = os.path.join(
        base_dir,
        "tab_data/uci_diabetes/all_train_baseline_uci_160k_40epoch.json")

    data_dict = read_json(mlp_train_uci_diabetes)
elif dataset_used == "criteo":
    mlp_train_criteo = os.path.join(
        base_dir,
        "tab_data/criteo/all_train_baseline_criteo.json")

    data_dict = read_json(mlp_train_criteo)
else:
    print("err")

epoch_sampled = {"frappe": 13, "uci_diabetes": 0, "criteo": 9}
draw_edcf()

"""
uci_diabetes
epoch 0 medium 0.6269790217702936 best 0.6788386421503422
epoch 1 medium 0.6486151000917154 best 0.6865065126914073
epoch 4 medium 0.6482881949777892 best 0.6912332509923426
epoch 5 medium 0.642950699527092 best 0.6922250195733658
epoch 6 medium 0.6369747652547154 best 0.6912801990144214
epoch 7 medium 0.6310640706471495 best 0.6900225134569015
epoch 8 medium 0.6253744970145643 best 0.689073196927341

frappe
epoch 13 medium 0.9772853248010285 best 0.9814130102767649
frappe
epoch 19 medium 0.9767004372606147 best 0.981358559726915

criteo
epoch 9 medium 0.801413788910975 best 0.8033541930059981
"""