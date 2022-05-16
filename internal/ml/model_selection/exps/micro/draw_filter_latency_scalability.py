import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FuncFormatter
from brokenaxes import brokenaxes

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# plt.rcParams['text.usetex'] = True
# # Set the font to resemble LaTeX's default style
# plt.rcParams['font.family'] = 'serif'


def thousands_formatter(x, pos):
    if x >= 1e3:
        return '{:.0f}k'.format(x * 1e-3)
    else:
        return '{:.1f}'.format(x)


thousands_format = FuncFormatter(thousands_formatter)


def read_a_file(file_path):
    try:
        with open(file_path) as f:
            data_in_db = json.load(f)
    except:
        return {}
    return data_in_db


# Helper function to load data
def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


# Set your plot parameters
bar_width = 0.35
opacity = 0.8
set_font_size = 15  # Set the font size
set_lgend_size = 12
set_tick_size = 12
colors = ['#729ECE', '#FFB579', '#98DF8A']
hatches = ['/', '\\', 'x', '//', '\\\\', 'xx', 'oo', 'OO', '..', '**']

# here run 10k rows for inference.,
# each sub-list is "compute time" and "data fetch time"

# Collecting data for plotting
frappe_datasets_result = {
    r"500 $m$": {
        'out-DB-cpu': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_scale/'
                      '/time_score_mlp_sp_criteo_batch_size_64_cpu_jacflow_500.json',
        'In-Db-opt': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_indb_scale/'
                     '/time_score_mlp_sp_criteo_batch_size_32_cpu_jacflow_500.json',
    },

    r"1k $m$": {'out-DB-cpu': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_scale/'
                              '/time_score_mlp_sp_criteo_batch_size_64_cpu_jacflow_1k.json',
                'In-Db-opt': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_indb_scale/'
                             '/time_score_mlp_sp_criteo_batch_size_32_cpu_jacflow_1k.json'},
    r"2k $m$": {'out-DB-cpu': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_scale/'
                              '/time_score_mlp_sp_criteo_batch_size_64_cpu_jacflow_2k.json',
                'In-Db-opt': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_indb_scale/'
                             '/time_score_mlp_sp_criteo_batch_size_32_cpu_jacflow_2k.json'},

    r"4k $m$": {'out-DB-cpu': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_scale/'
                              '/time_score_mlp_sp_criteo_batch_size_64_cpu_jacflow_4k.json',
                'In-Db-opt': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_indb_scale/'
                             '/time_score_mlp_sp_criteo_batch_size_32_cpu_jacflow_4k.json'},

}

datasets = list(frappe_datasets_result.keys())

# Plotting
fig, ax = plt.subplots(figsize=(6.4, 4.1))

# Initial flags to determine whether the labels have been set before
set_label_baseline_sys = True
set_ouy_system = True
set_label_in_db_data_preprocess = True
set_label_in_db_data_compute = True
set_label_in_db_data_others = True

baseline_sys_x_array = []
baseline_sys_y_array = []

sams_sys_x_array = []
sams_sys_y_array = []

set_label_in_db_data_query = True
set_label_in_db_data_copy_start_py = True
set_label_in_db_data_preprocess = True
set_label_in_db_data_compute = True
set_label_in_db_data_others = True
set_label_in_db_model_load = True

indices = []
index = 0
for dataset, valuedic in frappe_datasets_result.items():
    print(dataset)
    indices.append(index)

    indb_med_opt = read_a_file(valuedic["In-Db-opt"])
    outcpudb_med = read_a_file(valuedic["out-DB-cpu"])

    # set labesl
    label_baseline_sys = 'INDICES (seperate)' if set_label_baseline_sys else None
    ouy_system = 'INDICES' if set_ouy_system else None

    # set labesl
    label_in_db_model_load = 'Model Initialization' if set_label_in_db_model_load else None
    label_in_db_data_query = 'Data Retrieval & Preprocessing' if set_label_in_db_data_query else None
    label_in_db_data_copy_start_py = 'Data Copying' if set_label_in_db_data_copy_start_py else None
    label_in_db_data_preprocess = 'Data Preprocessing' if set_label_in_db_data_preprocess else None
    label_in_db_data_compute = 'JacFlow Computation' if set_label_in_db_data_compute else None
    label_in_db_data_others = 'Others' if set_label_in_db_data_others else None
    #
    in_db_data_query = sum(indb_med_opt['track_io_data_retrievel'][2:]) + sum(
        indb_med_opt['track_io_data_preprocess'][2:])
    in_db_data_model_load = sum(outcpudb_med['track_io_model_init'][2:])
    # here we only use one metrics in practice
    in_db_data_compute = sum(outcpudb_med['track_compute'][2:]) * 0.63

    # in-db with optimizization
    ax.bar(index + bar_width / 2, in_db_data_compute, bar_width, color=colors[2], hatch=hatches[2], zorder=2, alpha=opacity,
           label=label_in_db_data_compute, edgecolor='black')

    ax.bar(index + bar_width / 2, in_db_data_model_load, bar_width, color=colors[1], hatch=hatches[1], zorder=2, alpha=opacity,
           label=label_in_db_model_load,
           bottom=in_db_data_compute,
           edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2, alpha=opacity,
           label=label_in_db_data_query,
           bottom=in_db_data_model_load + in_db_data_compute,
           edgecolor='black')

    sams_sys_x_array.append(index + bar_width / 2)
    sams_sys_y_array.append(
        in_db_data_model_load + in_db_data_query + in_db_data_compute)

    # # out-db CPU
    out_db_data_query = sum(outcpudb_med['track_io_data_retrievel'][2:]) + sum(
        outcpudb_med['track_io_data_preprocess'][2:])
    out_db_data_model_load = sum(outcpudb_med['track_io_model_init'][2:])
    out_db_data_compute = sum(outcpudb_med['track_compute'][2:]) * 0.63
    out_db_data_preprocess = 0

    ax.bar(index - bar_width / 2, out_db_data_compute, bar_width, color=colors[2], hatch=hatches[2], zorder=2, alpha=opacity,
           edgecolor='black')

    ax.bar(index - bar_width / 2, out_db_data_model_load, bar_width, color=colors[1], hatch=hatches[1], zorder=2, alpha=opacity,
           edgecolor='black',
           bottom=out_db_data_compute,
           )
    ax.bar(index - bar_width / 2, out_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2, alpha=opacity,
           bottom=out_db_data_model_load + out_db_data_compute,
           edgecolor='black')

    baseline_sys_x_array.append(index - bar_width / 2)
    baseline_sys_y_array.append(
        out_db_data_model_load + out_db_data_query + out_db_data_preprocess + out_db_data_compute)

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_in_db_data_query = False
    set_label_in_db_data_copy_start_py = False
    set_label_in_db_data_preprocess = False
    set_label_in_db_data_compute = False
    set_label_in_db_data_others = False
    set_label_in_db_model_load = False

    index += 1

ax.plot(sams_sys_x_array, sams_sys_y_array, color='red', marker='*', linewidth=2)  # 'o' will add a marker at each point
ax.plot(baseline_sys_x_array, baseline_sys_y_array, color='green', marker='o',
        linewidth=2)  # 'o' will add a marker at each point

# measure the speedups
for i in range(len(sams_sys_x_array)):
    t1 = sams_sys_y_array[i]
    t2 = baseline_sys_y_array[i]
    print("Speedups = ", t2 / t1)

for i in range(len(sams_sys_x_array) - 1):
    print("sams_increase = ", sams_sys_y_array[i + 1] / sams_sys_y_array[i])
    print("baseline_increasse = ", baseline_sys_y_array[i + 1] / baseline_sys_y_array[i])

# legned etc
ax.set_ylabel(".", fontsize=25, color='white')
fig.text(0.001, 0.5, 'Breakdown Total Time \n of Filtering Phase (s)', va='center', rotation='vertical', fontsize=18)

# ax.set_ylim(top=20)

ax.set_xticks(indices)
ax.set_xticklabels(datasets, rotation=0, fontsize=24)

# ax.legend(fontsize=set_lgend_size - 2, ncol=2, )
ax.legend(fontsize=12, ncol=1, loc='upper left')

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
ax.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_selection/exp_result/macro_data_scale_sys.pdf")
fig.savefig(f"./internal/ml/model_selection/exp_result/macro_data_scale_sys.pdf",
            bbox_inches='tight')
