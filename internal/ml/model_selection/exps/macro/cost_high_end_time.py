from src.common.constant import Config
from src.query_api.query_api_mlp import GTMLP
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def measure_total_time_usage(t1, t2, dataset):
    """
    # uci, minis
    training_bsaed_time = 40.226666666667974
    time_budget = 5.510345879931426

    # frappe, minis
    training_bsaed_time = 1099.49959
    time_budget = 135.28362353441455

    # criteo, minis
    training_bsaed_time = 3299.0397362709045
    time_budget = 135.28362353441455
    """
    U = 1

    if dataset == Config.Criteo:
        N = 1800
        K = 18
        phase2_time = 6750
        phase2_epoch = phase2_time / 125

    elif dataset == Config.Frappe:
        N = 64700
        K = 647
        phase2_time = 6694.799999999999
        phase2_epoch = phase2_time / 2.8
    else:
        # this is uci
        N = 11300
        K = 113
        phase2_time = 158.2
        phase2_epoch = phase2_time / 1.4

    return 0, phase2_time, phase2_epoch


result = {}
print("\n-----------------------------\n")
for dataset in [Config.Criteo, Config.Frappe, Config.UCIDataset]:
    gtapi = GTMLP(dataset)
    t1 = gtapi.get_score_one_model_time("cpu")
    t2 = gtapi.get_train_one_epoch_time("gpu")
    s1, s2, s2epoch = measure_total_time_usage(t1, t2, dataset)
    result[dataset] = s1 + s2
    print(f"--hybrid , Dataset={dataset}, s1={s1}, s2={s2}, total={s1 + s2}")
print(result)

# lines' mark size
set_marker_size = 10
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 20
set_lgend_size = 15
set_tick_size = 15
frontinsidebox = 23

training_based_time_usage = {
    Config.Criteo: 3299.0397362709045 * 60,
    Config.Frappe: 1099.49959 * 60,
    Config.UCIDataset: 40.226666666667974 * 60,
}

training_based_Flops_usage = {
    Config.Criteo: 3299.0397362709045 * 60,
    Config.Frappe: 1099.49959 * 60,
    Config.UCIDataset: 40.226666666667974 * 60,
}

time_usage = {}
datasets = [Config.Criteo, Config.Frappe, Config.UCIDataset]
for dataset in datasets:
    # training-based GPU time usage, \system phase2 time usage only.
    time_usage[dataset] = [training_based_time_usage[dataset], result[dataset]]
    print(f"SpeedUp = {training_based_time_usage[dataset] / result[dataset]}, "
          f"train-ms = {training_based_time_usage[dataset]}, "
          f"train_free_ms = {result[dataset]}")

num_datasets = len(datasets)
num_bars = 2  # training-based, trails

bar_width = 0.25
opacity = 0.8

index = np.arange(num_datasets)

fig, ax = plt.subplots(figsize=(6.4, 4))
ax.grid()

colors = ['#729ECE', '#FFB579', '#98DF8A']  # Softer colors #FF7F7F
hatches = ['/', '\\', 'x', 'o', 'O', '.', '*']

# Set the font size
fontsize = set_font_size

# Plot bars for CPU-only
rects1 = ax.bar(index - bar_width, [time_usage[dataset][0] for dataset in datasets], bar_width,
                alpha=opacity, color=colors[0], hatch=hatches[0], edgecolor='black', label='Training-Based MS')

# Plot bars for GPU-only
rects2 = ax.bar(index, [time_usage[dataset][1] for dataset in datasets], bar_width,
                alpha=opacity, color=colors[1], hatch=hatches[1], edgecolor='black', label='TRAILS')

ax.set_ylabel('GPU time (s)', fontsize=fontsize)
ax.set_xticks(index - bar_width / 2)
ax.set_xticklabels(["Criteo", "Frappe", "Diabetes"], fontsize=fontsize)
ax.set_yscale('log')  # Set y-axis to logarithmic scale
# ax.legend(fontsize=fontsize)
ax.set_ylim(ymax=10 ** 6 * 2)
ax.legend(fontsize=fontsize-2, loc='upper right')

# Set the font size for tick labels
ax.tick_params(axis='both', labelsize=fontsize)
fig.tight_layout()
plt.tight_layout()

# export_legend(ori_fig=fig, colnum=5)
print(f"./cost_compwith_train_based.pdf")
fig.savefig("./cost_compwith_train_based.pdf", bbox_inches='tight')
