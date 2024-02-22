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
colors = ['#729ECE', '#8E44AD', '#2ECC71', '#3498DB', '#F39C12']
hatches = ['/', '\\', 'x', '.', '*', '//', '\\\\', 'xx', '..', '**']


def scale_to_ms(latencies):
    result = {}
    for key, value in latencies.items():
        value = value * 1000
        result[key] = value
    return result


# here run 10k rows for inference.,
# each sub-list is "compute time" and "data fetch time"
datasets_result = {
    'Adult': {
        'In-Db-opt': {'diff': -0.007969393999999852, 'mem_allocate_time': 0.007871151, 'model_init_time': 0.006836045,
                      'data_query_time': 1.4322866, 'python_compute_time': 0.624381187,
                      'overall_query_latency': 2.071473226, 'data_query_time_spi': 0.077206818,
                      'py_conver_to_tensor': 0.021612468719482422, 'py_compute': 0.4987836265563965,
                      'py_overall_duration': 0.49993324279785156, 'py_diff': 0.010537147521972656},

        'out-DB-cpu': {'data_query_time': 0.6272451877593994, 'py_conver_to_tensor': 0.0235938835144043,
                       'tensor_to_gpu': 0.00011944770812988281, 'py_compute': 0.49167609214782715,
                       'load_model': 0.12756085395812988, 'overall_query_latency': 1.254258155822754},
    },

    'Disease': {
        'In-Db-opt': {'diff': -0.0067551480000001, 'data_query_time_spi': 0.073574947, 'model_init_time': 0.006799476,
                      'mem_allocate_time': 0.006654889, 'python_compute_time': 1.045942085,
                      'data_query_time': 0.71951581, 'overall_query_latency': 1.7790125190000001,
                      'py_conver_to_tensor': 0.025516510009765625, 'py_compute': 0.8286852836608887,
                      'py_overall_duration': 0.8644576072692871, 'py_diff': 0.010255813598632812},

        'out-DB-cpu': {'data_query_time': 0.5226991176605225, 'py_conver_to_tensor': 0.02484127998352051,
                       'tensor_to_gpu': 0.00022840499877929688, 'py_compute': 0.8290549659729004,
                       'load_model': 0.1251354217529297, 'overall_query_latency': 1.657167673110962},
    },

    'Bank': {
        'In-Db-opt': {'data_query_time_spi': 0.068415319, 'model_init_time': 0.006328437,
                      'data_query_time': 1.780908539, 'mem_allocate_time': 0.009163106,
                      'python_compute_time': 0.715492727, 'overall_query_latency': 2.511971514,
                      'diff': -0.009241810999999878, 'py_conver_to_tensor': 0.02403855323791504,
                      'py_compute': 0.8074800453186035, 'py_overall_duration': 0.5948076248168945,
                      'py_diff': 0.009289026260375977},
        'out-DB-cpu': {'data_query_time': 0.7120261192321777, 'py_conver_to_tensor': 0.02698847770690918,
                       'tensor_to_gpu': 0.00021409988403320312, 'py_compute': 0.8072443008422852,
                       'load_model': 0.046822547912597656, 'overall_query_latency': 1.637943983078003},
    },

    'AppRec': {
        'In-Db-opt':
            {'data_query_time_spi': 0.077303043, 'python_compute_time': 1.083512117,
             'overall_query_latency': 1.618568955, 'diff': -0.006197264000000091, 'model_init_time': 0.007554649,
             'mem_allocate_time': 0.006098022, 'data_query_time': 0.521304925,
             'py_conver_to_tensor': 0.014301300048828125, 'py_compute': 0.900803804397583,
             'py_overall_duration': 0.924717903137207, 'py_diff': 0.009612798690795898},
        'out-DB-cpu':
            {'data_query_time': 0.5804119110107422, 'py_conver_to_tensor': 0.01865216255187988,
             'tensor_to_gpu': 0.0001499652862548828, 'py_compute': 0.9637076377868652,
             'load_model': 0.16712236404418945, 'overall_query_latency': 2.086137056350708},
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
for dataset, valuedic in datasets_result.items():
    indices.append(index)

    indb_med_opt = scale_to_ms(valuedic["In-Db-opt"])
    outcpudb_med = scale_to_ms(valuedic["out-DB-cpu"])

    # set labesl
    label_in_db_model_load = 'Model Loading' if set_label_in_db_model_load else None
    label_in_db_data_query = 'Data Retrieval' if set_label_in_db_data_query else None
    label_in_db_data_copy_start_py = 'Data Copying' if set_label_in_db_data_copy_start_py else None
    label_in_db_data_preprocess = 'Data Preprocessing' if set_label_in_db_data_preprocess else None
    label_in_db_data_compute = 'Inference' if set_label_in_db_data_compute else None
    label_in_db_data_others = 'Others' if set_label_in_db_data_others else None

    # in-db with optimizization
    in_db_data_model_load = indb_med_opt["model_init_time"]
    in_db_data_copy_start_py = indb_med_opt["python_compute_time"] - indb_med_opt["py_overall_duration"]
    in_db_data_query = indb_med_opt["data_query_time_spi"]
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
           bottom=in_db_data_query + in_db_data_model_load,
           label=label_in_db_data_copy_start_py,
           edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_preprocess + in_db_data_compute, bar_width, color=colors[2],
           hatch=hatches[2], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_model_load,
           label=label_in_db_data_preprocess, edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess + in_db_data_model_load,
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
           bottom=out_db_data_query + in_db_data_model_load,
           edgecolor='black')
    ax.bar(index - bar_width / 2, out_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=out_db_data_query + in_db_data_model_load + out_db_data_preprocess,
           edgecolor='black')

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_in_db_data_query = False
    set_label_in_db_data_copy_start_py = False
    set_label_in_db_data_preprocess = False
    set_label_in_db_data_compute = False
    set_label_in_db_data_others = False
    set_label_in_db_model_load = False

    index += 1

# legned etc
ax.set_ylabel(".", fontsize=20, color='white')
fig.text(0.01, 0.5, 'Response Time (ms)', va='center', rotation='vertical', fontsize=20)

ax.set_ylim(top=2200)

ax.set_xticks(indices)
ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)

# ax.legend(fontsize=set_lgend_size - 2, ncol=2, )
ax.legend(fontsize=set_lgend_size, ncol=2, loc='upper left')

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
ax.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/exp_imgs/int_format/macro.pdf")
fig.savefig(f"./internal/ml/model_slicing/exp_imgs/int_format/macro.pdf",
            bbox_inches='tight')
