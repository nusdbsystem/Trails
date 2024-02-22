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
bar_width = 0.35
set_font_size = 15
set_lgend_size = 12
set_tick_size = 12
colors = ['#8E44AD', '#729ECE', '#F39C12', '#2ECC71', '#3498DB', '#E74C3C', '#2C3E50', '#27AE60', '#F1C40F', '#9B59B6']
hatches = ['/', '\\', 'x', '.', '*', '//', '\\\\', 'xx', '..', '**']


def scale_to_ms(latencies):
    result = {}
    for key, value in latencies.items():
        value = value * 1000
        result[key] = value
    return result


# Data
datasets_result = {

    'INDICES w/ all optimizations':
        {'data_query_time_spi': 0.077303043, 'python_compute_time': 1.083512117,
         'overall_query_latency': 1.618568955, 'diff': -0.006197264000000091,
         'model_init_time': 0.007554649,
         'mem_allocate_time': 0.006098022, 'data_query_time': 0.521304925,
         'py_conver_to_tensor': 0.014301300048828125, 'py_compute': 0.900803804397583,
         'py_overall_duration': 0.924717903137207, 'py_diff': 0.009612798690795898},

    'INDICES w/o model caching':
        {'data_query_time_spi': 0.077303043, 'python_compute_time': 1.083512117,
         'overall_query_latency': 1.618568955, 'diff': -0.006197264000000091,
         'model_init_time': 0.16712236404418945,
         'mem_allocate_time': 0.006098022, 'data_query_time': 0.521304925,
         'py_conver_to_tensor': 0.014301300048828125, 'py_compute': 0.900803804397583,
         'py_overall_duration': 0.924717903137207, 'py_diff': 0.009612798690795898},

    'INDICES w/o memory sharing':
        {'data_query_time_spi': 0.077303043, 'python_compute_time': 1.083512117,
         'overall_query_latency': 1.618568955, 'diff': -0.006197264000000091,
         'model_init_time': 0.007554649,
         'mem_allocate_time': 0.006098022, 'data_query_time': 0.521304925,
         'py_conver_to_tensor': 0.014301300048828125, 'py_compute': 0.900803804397583,
         'py_overall_duration': 1.083512117 - 0.33, 'py_diff': 0.009612798690795898},

    'INDICES w/o SPI':
        {'data_query_time_spi': 0.5804119110107422, 'python_compute_time': 0.924717903137207,
         'overall_query_latency': 1.618568955, 'diff': -0.006197264000000091,
         'model_init_time': 0.007554649,
         'mem_allocate_time': 0.006098022, 'data_query_time': 0.521304925,
         'py_conver_to_tensor': 0.014301300048828125, 'py_compute': 0.900803804397583,
         'py_overall_duration': 0.924717903137207, 'py_diff': 0.009612798690795898},

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
    in_db_data_query = indb_med_opt["data_query_time_spi"]
    in_db_data_copy_start_py = indb_med_opt["python_compute_time"] - indb_med_opt["py_overall_duration"]
    in_db_data_preprocess = indb_med_opt["py_conver_to_tensor"]
    in_db_data_compute = indb_med_opt["py_compute"]
    in_db_data_others = indb_med_opt["py_diff"]

    # Vertical bars
    ax.barh(index, in_db_data_model_load, bar_height, color=colors[0], hatch=hatches[0], label=label_in_db_model_load,
            edgecolor='black', )
    ax.barh(index, in_db_data_query, bar_height, color=colors[1], hatch=hatches[1], label=label_in_db_data_query,
            left=in_db_data_model_load, edgecolor='black', )
    ax.barh(index, in_db_data_copy_start_py, bar_height, color=colors[2], hatch=hatches[2],
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

    index += 0.6

ax.set_yticks(indices)
ax.set_yticklabels(datasets, fontsize=20)

ax.xaxis.set_major_formatter(thousands_format)
ax.tick_params(axis='x', which='major', labelsize=20)
ax.set_xlabel('End-to-end Time (ms)', fontsize=20)

# Add legend
# ax.legend(fontsize=set_lgend_size, loc='upper left', ncol=5)
ax.legend(fontsize=15,
          loc='center', ncol=6,
          bbox_to_anchor=(0.22, 1.1))

# Grid and save the figure
ax.xaxis.grid(True)
plt.tight_layout()
fig.tight_layout()
plt.savefig("./internal/ml/model_slicing/exp_imgs/int_format/micro2.pdf", bbox_inches='tight')
