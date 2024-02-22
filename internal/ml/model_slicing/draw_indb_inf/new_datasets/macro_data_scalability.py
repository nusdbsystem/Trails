import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FuncFormatter
from brokenaxes import brokenaxes


def thousands_formatter(x, pos):
    if x >= 1e3:
        return '{:.0f}k'.format(x * 1e-3)
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

# Collecting data for plotting
frappe_datasets_result = {

    "40k": {'out-DB-cpu': {'load_model': 0.08967399597167969, 'data_query_time': 0.43170952796936035,
                           'py_conver_to_tensor': 0.051702022552490234, 'tensor_to_gpu': 3.0517578125e-05,
                           'py_compute': 0.07541322708129883, 'overall_query_latency': 0.6202089786529541},
            'In-Db-opt': {'data_query_time_spi': 0.027747653, 'python_compute_time': 0.150434476,
                          'data_type_convert_time': 0.194070956, 'overall_query_latency': 0.388566718,
                          'model_init_time': 0.010059603, 'data_query_time': 0.224520451,
                          'mem_allocate_time': 0.003547337, 'diff': -0.003552187999999956,
                          'py_conver_to_tensor': 0.009652853012084961, 'py_compute': 0.11598563194274902,
                          'py_overall_duration': 0.1287381649017334, 'py_diff': 0.003099679946899414}},

    "80k": {'out-DB-cpu': {'load_model': 0.08788943290710449, 'data_query_time': 0.9318385124206543,
                           'py_conver_to_tensor': 0.10923171043395996, 'tensor_to_gpu': 5.507469177246094e-05,
                           'py_compute': 0.1265401840209961, 'overall_query_latency': 1.2477145195007324},
            'In-Db-opt': {'overall_query_latency': 0.659449067, 'model_init_time': 0.00946634,
                          'mem_allocate_time': 0.007959983, 'data_query_time': 0.389034806,
                          'data_query_time_spi': 0.047165429, 'data_type_convert_time': 0.334752224,
                          'diff': -0.007966999999999946, 'python_compute_time': 0.252980921,
                          'py_conver_to_tensor': 0.016394853591918945, 'py_compute': 0.20447969436645508,
                          'py_overall_duration': 0.2266526222229004, 'py_diff': 0.005778074264526367}},

    "160k": {'out-DB-cpu': {'load_model': 0.0875093936920166, 'data_query_time': 1.7526452541351318,
                            'py_conver_to_tensor': 0.24686884880065918, 'tensor_to_gpu': 6.031990051269531e-05,
                            'py_compute': 0.2105252742767334, 'overall_query_latency': 2.335653781890869},
             'In-Db-opt': {'mem_allocate_time': 0.015960274, 'python_compute_time': 0.461230493,
                           'data_type_convert_time': 0.67199331, 'data_query_time': 0.782029675,
                           'overall_query_latency': 1.262588026, 'data_query_time_spi': 0.094563935,
                           'diff': -0.015967349999999936, 'model_init_time': 0.003360508,
                           'py_conver_to_tensor': 0.01498723030090332, 'py_compute': 0.3437178134918213,
                           'py_overall_duration': 0.38239526748657227, 'py_diff': 0.023690223693847656}},

    "320k": {'out-DB-cpu': {'load_model': 0.08763766288757324, 'data_query_time': 3.358307361602783,
                            'py_conver_to_tensor': 0.4489321708679199, 'tensor_to_gpu': 6.198883056640625e-05,
                            'py_compute': 0.8973984718322754, 'overall_query_latency': 4.937885046005249},
             'In-Db-opt': {'diff': -0.03916125699999995, 'overall_query_latency': 2.4579261,
                           'data_type_convert_time': 1.354739719, 'python_compute_time': 0.824346779,
                           'data_query_time': 1.584931535, 'model_init_time': 0.009486529,
                           'data_query_time_spi': 0.17950512, 'mem_allocate_time': 0.039150373,
                           'py_conver_to_tensor': 0.016873598098754883, 'py_compute': 0.6497747898101807,
                           'py_overall_duration': 0.6944558620452881, 'py_diff': 0.02780747413635254}},

    "640k": {
        'out-DB-cpu': {'load_model': 0.08840274810791016, 'data_query_time': 6.585202693939209,
                       'py_conver_to_tensor': 0.9350695610046387, 'tensor_to_gpu': 6.818771362304688e-05,
                       'py_compute': 1.6609573364257812, 'overall_query_latency': 9.600078344345093},
        'In-Db-opt': {'data_query_time_spi': 0.333654391, 'data_query_time': 3.145211651,
                      'model_init_time': 0.003357985, 'mem_allocate_time': 0.069457039,
                      'data_type_convert_time': 2.734944258, 'overall_query_latency': 4.9711342,
                      'python_compute_time': 1.7530989, 'diff': -0.06946566399999998,
                      'py_conver_to_tensor': 0.02354264259338379, 'py_compute': 1.4119174480438232,
                      'py_overall_duration': 1.5316107273101807, 'py_diff': 0.09615063667297363}},

}

datasets = list(frappe_datasets_result.keys())

# Plotting
fig, ax = plt.subplots(figsize=(6.4, 4.5))

# Initial flags to determine whether the labels have been set before
set_label_in_db_data_query = True
set_label_in_db_data_copy_start_py = True
set_label_in_db_data_preprocess = True
set_label_in_db_data_compute = True
set_label_in_db_data_others = True
set_label_in_db_model_load = True

baseline_sys_x_array = []
baseline_sys_y_array = []

sams_sys_x_array = []
sams_sys_y_array = []

indices = []
index = 0
for dataset, valuedic in frappe_datasets_result.items():
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

    sams_sys_x_array.append(index + bar_width / 2)
    sams_sys_y_array.append(
        in_db_data_model_load + in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess + in_db_data_compute)

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

    baseline_sys_x_array.append(index - bar_width / 2)
    baseline_sys_y_array.append(
        in_db_data_model_load + out_db_data_query + out_db_data_preprocess + out_db_data_compute)

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
ax.set_ylabel(".", fontsize=20, color='white')
fig.text(0.01, 0.5, 'Response Time (ms)', va='center', rotation='vertical', fontsize=20)

ax.set_ylim(top=9900)

ax.set_xticks(indices)
ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
ax.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_font_size)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
export_legend(
    fig,
    colnum=3,
    unique_labels=['Model Loading', 'Data Retrieval', 'Data Copying',
                   'Data Preprocessing', 'Inference'])

print(f"saving to ./internal/ml/model_slicing/exp_imgs/macro_data_scale.pdf")
fig.savefig(f"./internal/ml/model_slicing/exp_imgs/macro_data_scale_sys.pdf",
            bbox_inches='tight')
