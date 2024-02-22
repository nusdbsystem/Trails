import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def thousands_formatter(x, pos):
    if x >= 1e3:
        return '{:.1f}k'.format(x * 1e-3)
    else:
        return '{:.1f}'.format(x)


thousands_format = FuncFormatter(thousands_formatter)

# Set your plot parameters
from config import *

def scale_to_ms(latencies):
    result = {}
    for key, value in latencies.items():
        value = value * 1000
        result[key] = value
    return result

color_dic = {
    "Model Loading": colors[4],
    "Data Retrieval": colors[0],
    "Data Copying": colors[1],
    "Data Preprocessing & Inference": colors[3],
}


# Data
datasets_result = {

    'INDICES w/ all optimizations':
        {'data_type_convert_time': 0.425379366, 'overall_query_latency': 0.813545531,
         'mem_allocate_time': 0.009849753, 'data_query_time': 0.499067735, 'model_init_time': 0.007176829,
         'data_query_time_spi': 0.065228678, 'python_compute_time': 0.297439744, 'diff': -0.00986122300000003,
         'py_conver_to_tensor': 0.014742374420166016, 'py_compute': 0.24869871139526367,
         'py_overall_duration': 0.27048182487487793, 'py_diff': 0.007040739059448242},


    'INDICES w/o memory sharing':
        {"mem_allocate_time": 0, 'model_init_time': 0.005273089, 'diff': -6.108000000004665e-06,
         'data_query_time_spi': 0.062182906, 'data_query_time': 0.504550241, 'overall_query_latency': 2.263737131,
         'python_compute_time': 1.6939076929999999, 'py_conver_to_tensor': 0.125929594039917,
         'py_compute': 0.2513298683166504, 'py_overall_duration': 1.5709002017974854, 'py_diff': 1.186640739440918},

    'INDICES w/o SPI':
        {'data_type_convert_time': 0.425379366, 'overall_query_latency': 0.813545531,
         'mem_allocate_time': 0.009849753, 'data_query_time': 1.1151514053344727, 'model_init_time': 0.007176829,
         'data_query_time_spi': 0.065228678, 'python_compute_time': 0.297439744, 'diff': -0.00986122300000003,
         'py_conver_to_tensor': 0.014742374420166016, 'py_compute': 0.24869871139526367,
         'py_overall_duration': 0.27048182487487793, 'py_diff': 0.007040739059448242},

    'INDICES w/o state caching':
        {'python_compute_time': 0.321392947, 'data_query_time_spi': 0.053119287, 'overall_query_latency': 1.916763343,
         'mem_allocate_time': 0.009899045, 'data_type_convert_time': 0.418267776, 'diff': -0.009906367999999777,
         'data_query_time': 0.500099237, 'model_init_time': 1.105364791, 'py_conver_to_tensor': 0.01148843765258789,
         'py_compute': 0.2511436939239502, 'py_overall_duration': 0.2696952819824219, 'py_diff': 0.007063150405883789},

    'INDICES w/o all optimizations':
        {'data_type_convert_time': 0.425379366, 'overall_query_latency': 0.813545531,
         'mem_allocate_time': 0.009849753, 'data_query_time': 1.1151514053344727, 'model_init_time':  1.105364791,
         'data_query_time_spi': 0.065228678, 'python_compute_time': 1.6939076929999999, 'diff': -0.00986122300000003,
         'py_conver_to_tensor': 0.014742374420166016, 'py_compute': 0.24869871139526367,
         'py_overall_duration': 1.5709002017974854, 'py_diff': 0.007040739059448242}

}

datasets = list(datasets_result.keys())

# Plotting
fig, ax = plt.subplots(figsize=(12, 3.8))

# Initial flags to determine whether the labels have been set before
set_label_in_db_model_load = True
set_label_in_db_data_query = True
set_label_in_db_data_copy_start_py = True
set_label_in_db_data_preprocess = True
set_label_in_db_data_compute = True
set_label_in_db_data_others = True

indices = []
index = 0
bar_height = 0.35

for dataset, valuedic in datasets_result.items():
    indb_med_opt = scale_to_ms(valuedic)
    indices.append(index)
    # set labels
    label_in_db_model_load = 'Model Loading' if set_label_in_db_model_load else None
    label_in_db_data_query = 'Data Retrieval' if set_label_in_db_data_query else None
    label_in_db_data_copy_start_py = 'Data Copying' if set_label_in_db_data_copy_start_py else None
    label_in_db_data_preprocess = 'Data Preprocessing & Inference' if set_label_in_db_data_preprocess else None
    # label_in_db_data_compute = 'Inference' if set_label_in_db_data_compute else None
    label_in_db_data_others = 'Others' if set_label_in_db_data_others else None

    # in-db with optimization
    in_db_data_model_load = indb_med_opt["model_init_time"]
    in_db_data_query = indb_med_opt["data_query_time"]
    in_db_data_copy_start_py = indb_med_opt["mem_allocate_time"] + indb_med_opt["python_compute_time"] - indb_med_opt[
        "py_overall_duration"]
    in_db_data_preprocess = indb_med_opt["py_conver_to_tensor"]
    in_db_data_compute = indb_med_opt["py_compute"]
    in_db_data_others = indb_med_opt["py_diff"]

    # Vertical bars
    ax.barh(index, in_db_data_model_load, bar_height, color=colors[4], hatch=hatches[4], label=label_in_db_model_load,
            edgecolor='black', )
    ax.barh(index, in_db_data_query, bar_height, color=colors[0], hatch=hatches[0], label=label_in_db_data_query,
            left=in_db_data_model_load, edgecolor='black', )
    ax.barh(index, in_db_data_copy_start_py, bar_height, color=colors[1], hatch=hatches[1],
            label=label_in_db_data_copy_start_py, left=in_db_data_query + in_db_data_model_load, edgecolor='black', )
    ax.barh(index, in_db_data_preprocess + in_db_data_compute, bar_height, color=colors[3], hatch=hatches[3],
            label=label_in_db_data_preprocess, left=in_db_data_query + in_db_data_copy_start_py + in_db_data_model_load,
            edgecolor='black', )
    # ax.barh(index, in_db_data_compute, bar_height, color=colors[4], hatch=hatches[4], label=label_in_db_data_compute, left=in_db_data_query+in_db_data_copy_start_py+in_db_data_preprocess+in_db_data_model_load, edgecolor='black', )
    # ax.barh(index, in_db_data_others, bar_height, color=colors[5], hatch=hatches[5], label=label_in_db_data_others, left=in_db_data_query+in_db_data_copy_start_py+in_db_data_preprocess+in_db_data_compute+in_db_data_model_load, edgecolor='black', )

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_in_db_model_load = False
    set_label_in_db_data_query = False
    set_label_in_db_data_copy_start_py = False
    set_label_in_db_data_preprocess = False
    set_label_in_db_data_compute = False
    set_label_in_db_data_others = False

    index += 0.8

ax.set_yticks(indices)
ax.set_yticklabels(datasets, fontsize=20)

ax.xaxis.set_major_formatter(thousands_format)
ax.tick_params(axis='x', which='major', labelsize=20)
ax.set_xlabel('Response Time (ms)', fontsize=20)

# Add legend
# ax.legend(fontsize=set_lgend_size, loc='upper left', ncol=5)
# ax.legend(fontsize=18,
#           loc='center', ncol=2,
#           bbox_to_anchor=(0.22, 1.2))

export_legend(
    fig,
    filename="micro2_leg",
    colnum=2,
    unique_labels=['Model Loading', 'Data Retrieval', 'Data Copying',
                   'Data Preprocessing & Inference'])

# Grid and save the figure
ax.xaxis.grid(True)
plt.tight_layout()
fig.tight_layout()
fig.savefig("./internal/ml/model_slicing/exp_imgs/micro2.pdf", bbox_inches='tight')
