from src.common.constant import Config
from src.query_api.query_api_mlp import GTMLP
import math

"""
Frappe in-db: avg_compute_latency=0.016814214079415336, avg_track_io_model_init=0.004127795622796237, avg_track_io_data_retrievel_preprocess=0.0022385170956810457, 
Frappe out-db: avg_compute_latency=0.02822492536710424, avg_track_io_model_init=0.004757730614523641, avg_track_io_data_retrievel_preprocess=0.0047602025257582285, 
============================================================
Diabete in-db: avg_compute_latency=0.02227595611876037, avg_track_io_model_init=0.005861111906539349, avg_track_io_data_retrievel_preprocess=0.003878541639586743, 
Diabete out-db: avg_compute_latency=0.021550353167771624, avg_track_io_model_init=0.005037131620531514, avg_track_io_data_retrievel_preprocess=0.008015334630022053, 
============================================================
Criteo in-db: avg_compute_latency=0.018855932475376626, avg_track_io_model_init=0.005302829664199054, avg_track_io_data_retrievel_preprocess=0.0033248917681544055, 
Criteo out-db: avg_compute_latency=0.02387025471733493, avg_track_io_model_init=0.0049122056754983484, avg_track_io_data_retrievel_preprocess=0.008181101801682586, 
"""


def measure_total_time_usage(t1, t2, dataset):
    N = 1500
    K = 10
    U = 1
    phase1_time = N * t1
    model_init_time = N * 0.00503
    if dataset == Config.UCIDataset:
        # only 1 epoch since fully training is 1 epoch only
        phase2_time = K * U * t2
    else:
        phase2_time = K * U * t2 * int(math.log(K, 3))

    return phase1_time + model_init_time, phase2_time


def get_data_loading_time(dataset, place, N=1500):
    if dataset == Config.Frappe:
        t_one_ite = 0.07835125923156738
        t_all = 5.960570335388184
        if place == "in-db":
            filter_phase_load_data = N * 0.0022385170956810457
        else:
            filter_phase_load_data = N * 0.0047602025257582285
    elif dataset == Config.Criteo:
        t_one_ite = 12.259164810180664
        t_all = 1814.5491938591003
        if place == "in-db":
            filter_phase_load_data = N * 0.0033248917681544055
        else:
            filter_phase_load_data = N * 0.008181101801682586
    elif dataset == Config.UCIDataset:
        t_one_ite = 0.11569786071777344
        t_all = 4.2008748054504395
        if place == "in-db":
            filter_phase_load_data = N * 0.003878541639586743
        else:
            filter_phase_load_data = N * 0.008015334630022053
    else:
        raise
    return t_one_ite, t_all, filter_phase_load_data


result = {}
print("\n-----------------------------\n")
for dataset in [Config.Criteo, Config.Frappe, Config.UCIDataset]:
    gtapi = GTMLP(dataset)
    t1 = gtapi.get_score_one_model_time("cpu")
    t2 = gtapi.get_train_one_epoch_time("gpu")
    s1, s2 = measure_total_time_usage(t1, t2, dataset)
    if dataset not in result:
        result[dataset] = []
    data_load_time_one_ite, data_load_time_one_all, in_filter = get_data_loading_time(dataset, "in-db")
    out_data_load_time_one_ite, out_data_load_time_one_all, out_filter = get_data_loading_time(dataset, "out-db")
    result[dataset].append(s1 + s2 + data_load_time_one_ite + in_filter)
    result[dataset].append(s1 + s2 + data_load_time_one_all + out_filter)
    print(f"--hybrid , Dataset={dataset}, s1={s1}, s2={s2}, total={s1 + s2}")
print(result)

import numpy as np
import matplotlib.pyplot as plt

# lines' mark size
set_marker_size = 10
# points' mark size
set_marker_point = 14
# points' mark size
set_font_size = 20
set_lgend_size = 15
set_tick_size = 15
frontinsidebox = 23

time_usage = {}
datasets = [Config.Criteo, Config.Frappe, Config.UCIDataset]
for dataset in datasets:
    time_usage[dataset] = result[dataset]

num_datasets = len(datasets)
num_bars = 3  # CPU-only, GPU-only, Hybrid

bar_width = 0.25
opacity = 0.8

index = np.arange(num_datasets)

fig, ax = plt.subplots(figsize=(6.4, 4.5))
ax.grid()

colors = ['#729ECE', '#FFB579', '#98DF8A']  # Softer colors #FF7F7F
hatches = ['/', '\\', 'x', 'o', 'O', '.', '*']

# Set the font size
fontsize = set_font_size

rects1 = ax.bar(index - bar_width, [time_usage[dataset][0] for dataset in datasets], bar_width,
                alpha=opacity, color=colors[0], hatch=hatches[0], edgecolor='black', label='TRAILS')

rects2 = ax.bar(index, [time_usage[dataset][1] for dataset in datasets], bar_width,
                alpha=opacity, color=colors[1], hatch=hatches[1], edgecolor='black', label='TRAILS (Decouple)')

ax.set_ylabel('Latency (s)', fontsize=fontsize)
ax.set_xticks(index)
ax.set_xticklabels(["Criteo", "Frappe", "Diabetes"], fontsize=fontsize)

# linear', 'log', 'symlog', 'logit', 'function', 'functionlog'
ax.set_yscale('symlog')  # Set y-axis to logarithmic scale

ax.set_ylim(ymax=10 ** 3 * 7)

ax.legend(fontsize=fontsize)

# Set the font size for tick labels
ax.tick_params(axis='both', labelsize=fontsize)
fig.tight_layout()
plt.tight_layout()

# export_legend(ori_fig=fig, colnum=5)

fig.savefig("./IDMS_latency.pdf", bbox_inches='tight')
