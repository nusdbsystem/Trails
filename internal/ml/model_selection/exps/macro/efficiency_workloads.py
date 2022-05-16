from src.common.constant import Config
from src.query_api.query_api_mlp import GTMLP
import math
from matplotlib.ticker import FuncFormatter


# Define the formatter function
def thousands_formatter(x, pos):
    'The two args are the value and tick position'
    if x >= 1000:
        return '{}k'.format(int(x * 1e-3))  # No decimal places needed
    return str(int(x))


def measure_total_time_usage(N, K, t1, t2, dataset):
    U = 1
    phase1_time = N * t1
    if dataset == "uci":
        # only 1 epoch since fully training is 1 epoch only
        phase2_time = K * U * t2
    else:
        phase2_time = K * U * t2 * int(math.log(K, 3))
    return phase1_time, phase2_time


def get_data_loading_time(dataset):
    if dataset == Config.Frappe:
        t_one_ite = 0.07835125923156738
        t_all = 5.960570335388184
    elif dataset == Config.Criteo:
        t_one_ite = 12.259164810180664
        t_all = 1814.5491938591003
    elif dataset == Config.UCIDataset:
        t_one_ite = 0.11569786071777344
        t_all = 4.2008748054504395
    else:
        raise
    return t_one_ite, t_all


gtapi = GTMLP(Config.Criteo)
t1 = gtapi.get_score_one_model_time("cpu")
t2 = gtapi.get_train_one_epoch_time("gpu")

result = {}
datasets = [[500, 5], [1000, 10], [2000, 20], [4000, 40]]
print("\n-----------------------------\n")
for nk_pair in datasets:
    if str(nk_pair) not in result:
        result[str(nk_pair)] = []
    N, K = nk_pair[0], nk_pair[1]
    s1, s2 = measure_total_time_usage(N, K, t1, t2, Config.Criteo)
    data_load_time_one_ite, data_load_time_one_all = get_data_loading_time(Config.Criteo)

    result[str(nk_pair)].append(s1 + s2 + data_load_time_one_ite)
    result[str(nk_pair)].append(s1 + s2 + data_load_time_one_all)
    print(f"{str(nk_pair)} speed ups = {(s1 + s2 + data_load_time_one_all)/(s1 + s2 + data_load_time_one_ite)}")
print(result)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# lines' mark size
set_marker_size = 10
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 25
set_lgend_size = 15
set_tick_size = 25
frontinsidebox = 23

time_usage = {}
for dataset in datasets:
    time_usage[str(dataset)] = result[str(dataset)]

num_datasets = len(datasets)
num_bars = 4  # CPU-only, GPU-only, Hybrid

bar_width = 0.25
opacity = 0.8

index = np.arange(num_datasets)

fig, ax = plt.subplots(figsize=(6.4, 3.7))
# ax.grid()

colors = ['#729ECE', '#FFB579', '#98DF8A', "#D1A7DC"]  # Softer colors #FF7F7F
hatches = ['/', '\\', 'x', 'o', 'O', '.', '*']

# Set the font size
fontsize = set_font_size

# Plot bars for external dataloader
rects1 = ax.bar(index - bar_width, [time_usage[str(dataset)][0] for dataset in datasets], bar_width,
                alpha=opacity, color=colors[0], hatch=hatches[0], edgecolor='black', label='TRAILS')

# Plot bars for IDMS
rects2 = ax.bar(index, [time_usage[str(dataset)][1] for dataset in datasets], bar_width,
                alpha=opacity, color=colors[1], hatch=hatches[1], edgecolor='black', label='TRAILS (Decoupled)')

ax.set_ylabel('Query Latency (s)', fontsize=fontsize)
# ax.set_xlabel('Explored Models $N$', fontsize=fontsize)
ax.set_xticks(index)
ax.set_xticklabels([r"500 $m$", r"1k $m$", r"2k $m$", r"4k $m$"], fontsize=fontsize)

# linear', 'log', 'symlog', 'logit', 'function', 'functionlog'
ax.set_yscale('linear')  # Set y-axis to logarithmic scale

ax.set_ylim(ymax=19999)

# yticks_positions = [1, 5000, 10000, 15000]
# yticks_labels = ['1', '5k', '10k', '15k']
# plt.yticks(yticks_positions, yticks_labels)

formatter = FuncFormatter(thousands_formatter)
ax.yaxis.set_major_formatter(formatter)

# ax.legend(fontsize=fontsize)

# Set the font size for tick labels
ax.tick_params(axis='both', labelsize=fontsize)
fig.tight_layout()
plt.tight_layout()
plt.grid()

# export_legend(ori_fig=fig, colnum=5)

plt.legend(fontsize=17, loc='upper left', ncol=1)
fig.savefig("./IDMS_latency_workloads.pdf", bbox_inches='tight')
