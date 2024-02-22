import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FuncFormatter
from brokenaxes import brokenaxes


def thousands_formatter(x, pos):
    if x >= 1e3:
        return '{:.1f}k'.format(x * 1e-3)
    else:
        return '{:.1f}'.format(x)


thousands_format = FuncFormatter(thousands_formatter)

from config import *

# Helper function to load data
def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def scale_to_ms(latencies):
    result = {}
    for key, value in latencies.items():
        value = value * 1000
        result[key] = value
    return result


# here run 10k rows for inference.,
# each sub-list is "compute time" and "data fetch time"
datasets_result = {

    # Credit datasets
    'Payment': {
        'In-Db-opt':
            {'data_type_convert_time': 0.425379366, 'overall_query_latency': 0.813545531,
             'mem_allocate_time': 0.009849753, 'data_query_time': 0.499067735, 'model_init_time': 0.007176829,
             'data_query_time_spi': 0.065228678, 'python_compute_time': 0.297439744, 'diff': -0.00986122300000003,
             'py_conver_to_tensor': 0.014742374420166016, 'py_compute': 0.24869871139526367,
             'py_overall_duration': 0.27048182487487793, 'py_diff': 0.007040739059448242},
        'out-DB-cpu':
            {'load_model': 0.0888833999633789, 'data_query_time': 1.1151514053344727,
             'py_conver_to_tensor': 0.12262678146362305, 'tensor_to_gpu': 0.0001277923583984375,
             'py_compute': 0.24600468635559082, 'overall_query_latency': 1.4884955883026123},
    },

    # Hcdr datasets
    'Credit': {
        'In-Db-opt': {'python_compute_time': 0.539859344, 'data_query_time_spi': 0.093642878,
                      'model_init_time': 0.009644935, 'data_type_convert_time': 1.195698354,
                      'mem_allocate_time': 0.037618466, 'diff': -0.03763291099999977,
                      'overall_query_latency': 1.9069439479999999, 'data_query_time': 1.319806758,
                      'py_conver_to_tensor': 0.014055728912353516, 'py_compute': 0.3534198532104492,
                      'py_overall_duration': 0.498582124710083, 'py_diff': 0.0071065425872802734},

        'out-DB-cpu': {'load_model': 0.09451985359191895, 'data_query_time': 2.8837993144989014,
                       'py_conver_to_tensor': 0.4155910015106201, 'tensor_to_gpu': 4.9591064453125e-05,
                       'py_compute': 0.35349082946777344, 'overall_query_latency': 3.9349310398101807},
    },

    # Census
    'Census': {
        'In-Db-opt': {'mem_allocate_time': 0.017783244, 'data_query_time': 0.793543811,
                      'data_query_time_spi': 0.066880971, 'model_init_time': 0.009516155, 'diff': -0.017790607000000014,
                      'python_compute_time': 0.463485706, 'data_type_convert_time': 0.717144931,
                      'overall_query_latency': 1.2843362790000001, 'py_conver_to_tensor': 0.0161130428314209,
                      'py_compute': 0.2696975326538086, 'py_overall_duration': 0.4337773323059082,
                      'py_diff': 0.007966756820678711},

        'out-DB-cpu': {'load_model': 0.09210062026977539, 'data_query_time': 1.8442931175231934,
                       'py_conver_to_tensor': 0.2480144500732422, 'tensor_to_gpu': 0.00010943412780761719,
                       'py_compute': 0.26346492767333984, 'overall_query_latency': 2.5028600692749023},
    },

    # Diabetes dataset
    'Diabetes': {
        'In-Db-opt': {'data_query_time': 0.9428635, 'mem_allocate_time': 0.028375419, 'diff': -0.02838857600000022,
                      'overall_query_latency': 1.4700126550000001, 'data_type_convert_time': 0.846978815,
                      'model_init_time': 0.00999991, 'python_compute_time': 0.488760669,
                      'data_query_time_spi': 0.072860661, 'py_conver_to_tensor': 0.019464969635009766,
                      'py_compute': 0.4201478958129883, 'py_overall_duration': 0.45311784744262695,
                      'py_diff': 0.013504981994628906},
        'out-DB-cpu': {'load_model': 0.09108734130859375, 'data_query_time': 2.0884439945220947,
                       'py_conver_to_tensor': 0.2983376979827881, 'tensor_to_gpu': 5.936622619628906e-05,
                       'py_compute': 0.4253865146636963, 'overall_query_latency': 2.9439947605133057},
    },


}

datasets = list(datasets_result.keys())

# Plotting
fig, ax = plt.subplots(figsize=(6.4, 4.5))

# Initial flags to determine whether the labels have been set before
set_label_in_db_data_query = True
set_label_in_db_data_copy_start_py = True
set_label_in_db_data_preprocess = True
set_label_in_db_data_compute = True
set_label_in_db_data_others = True
set_label_in_db_model_load = True

indices = []
index = 0
speed_up_list = []

for dataset, valuedic in datasets_result.items():
    indices.append(index)

    indb_med_opt = scale_to_ms(valuedic["In-Db-opt"])
    outcpudb_med = scale_to_ms(valuedic["out-DB-cpu"])

    speed_up_list.append([dataset, indb_med_opt["overall_query_latency"], outcpudb_med["overall_query_latency"]])

    # set labesl
    label_in_db_model_load = 'Model Loading' if set_label_in_db_model_load else None
    label_in_db_data_query = 'Data Retrieval' if set_label_in_db_data_query else None
    label_in_db_data_copy_start_py = 'Data Copying' if set_label_in_db_data_copy_start_py else None
    label_in_db_data_preprocess = 'Data Preprocessing' if set_label_in_db_data_preprocess else None
    label_in_db_data_compute = 'Inference' if set_label_in_db_data_compute else None
    label_in_db_data_others = 'Others' if set_label_in_db_data_others else None

    # in-db with optimizization
    in_db_data_model_load = indb_med_opt["model_init_time"]
    # this is rust_python_switch_time + python_read_from_shared_memory_time
    in_db_data_copy_start_py = indb_med_opt["mem_allocate_time"] + indb_med_opt["python_compute_time"] - indb_med_opt[
        "py_overall_duration"]
    in_db_data_query = indb_med_opt["data_query_time"]
    in_db_data_preprocess = indb_med_opt["py_conver_to_tensor"]
    in_db_data_compute = indb_med_opt["py_compute"]

    ax.bar(index + bar_width / 2, in_db_data_model_load, bar_width, color=colors[4], hatch=hatches[4], zorder=2,
           label=label_in_db_model_load,
           edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           label=label_in_db_data_query,
           bottom=in_db_data_model_load,
           edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_copy_start_py, bar_width, color=colors[1], hatch=hatches[1], zorder=2,
           bottom=in_db_data_model_load + in_db_data_query,
           label=label_in_db_data_copy_start_py,
           edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_preprocess, bar_width, color=colors[2],
           hatch=hatches[2], zorder=2,
           bottom=in_db_data_model_load + in_db_data_query + in_db_data_copy_start_py,
           label=label_in_db_data_preprocess, edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_model_load + in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess,
           label=label_in_db_data_compute, edgecolor='black')

    # # out-db CPU
    out_db_data_query = outcpudb_med["data_query_time"]
    out_db_data_preprocess = outcpudb_med["py_conver_to_tensor"]
    out_db_data_compute = outcpudb_med["py_compute"]

    ax.bar(index + bar_width / 2, in_db_data_model_load, bar_width, color=colors[4], hatch=hatches[4], zorder=2,
           edgecolor='black')
    ax.bar(index - bar_width / 2, out_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           bottom=in_db_data_model_load,
           edgecolor='black')
    ax.bar(index - bar_width / 2, out_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           bottom=in_db_data_model_load + out_db_data_query,
           edgecolor='black')
    ax.bar(index - bar_width / 2, out_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_model_load + out_db_data_query + out_db_data_preprocess,
           edgecolor='black')

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_in_db_data_query = False
    set_label_in_db_data_copy_start_py = False
    set_label_in_db_data_preprocess = False
    set_label_in_db_data_compute = False
    set_label_in_db_data_others = False
    set_label_in_db_model_load = False

    index += 1

# measure the speedups
for ele in speed_up_list:
    dataset = ele[0]
    in_db_t = ele[1]
    out_db_t = ele[2]
    print(f"{dataset}, Speedups = {out_db_t / in_db_t}")

# legned etc
ax.set_ylabel(".", fontsize=20, color='white')
fig.text(0.01, 0.5, 'Response Time (ms)', va='center', rotation='vertical', fontsize=20)

# ax.set_ylim(top=2200)

ax.set_xticks(indices)
ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)

# ax.legend(fontsize=set_lgend_size - 2, ncol=2, )
# ax.legend(fontsize=set_lgend_size, ncol=2, loc='upper left')

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
ax.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_font_size)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/exp_imgs/macro.pdf")
fig.savefig(f"./internal/ml/model_slicing/exp_imgs/macro.pdf",
            bbox_inches='tight')

"""
{
'overall_query_latency': 1.6173694090000001,
'diff': -0.018389589000000095,

'model_init_time': 0.009314114, 

'data_query_time': 0.781044081, 
'data_type_convert_time': 0.722756837, 
'data_query_time_spi': 0.046655084,

'mem_allocate_time': 0.018384798, 
'python_compute_time': 0.808621625, 

'py_conver_to_tensor': 0.005879878997802734, 

'py_compute': 0.7778224945068359,

'py_overall_duration': 0.8028597831726074, 
'py_diff': 0.01915740966796875}
"""
