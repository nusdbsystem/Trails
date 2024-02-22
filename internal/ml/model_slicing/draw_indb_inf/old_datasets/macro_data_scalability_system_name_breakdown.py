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

# Collecting data for plotting
frappe_datasets_result = {

    "40k": {'out-DB-cpu': {'data_query_time': 0.2667415142059326, 'py_conver_to_tensor': 0.022869348526000977,
                           'tensor_to_gpu': 0.00012683868408203125, 'py_compute': 0.37979626655578613,
                           'load_model': 0.1475677490234375, 'overall_query_latency': 0.811166524887085},
            'In-Db-opt': {'overall_query_latency': 0.858268554, 'python_compute_time': 0.541558046,
                          'data_query_time_spi': 0.033920884, 'data_query_time': 0.303039086,
                          'model_init_time': 0.008473502, 'mem_allocate_time': 0.005059687,
                          'diff': -0.005197919999999967, 'py_conver_to_tensor': 0.006430625915527344,
                          'py_compute': 0.40892887115478516, 'py_overall_duration': 0.4206066131591797,
                          'py_diff': 0.0052471160888671875},
            },

    "80k": {'out-DB-cpu': {'data_query_time': 0.5223848819732666, 'py_conver_to_tensor': 0.05331301689147949,
                           'tensor_to_gpu': 0.00040650367736816406, 'py_compute': 0.8352866172790527,
                           'load_model': 0.16507482528686523, 'overall_query_latency': 1.5977146625518799},
            'In-Db-opt': {'model_init_time': 0.008120983, 'python_compute_time': 0.794293001,
                          'overall_query_latency': 1.827405365, 'data_query_time_spi': 0.041727533,
                          'diff': -0.004288155999999876, 'data_query_time': 1.020703225,
                          'mem_allocate_time': 0.004241898, 'py_conver_to_tensor': 0.01320028305053711,
                          'py_compute': 0.60394287109375, 'py_overall_duration': 0.629204511642456,
                          'py_diff': 0.012061357498168945}, },
    "160k": {'out-DB-cpu': {'data_query_time': 0.9279088973999023, 'py_conver_to_tensor': 0.08997559547424316,
                            'tensor_to_gpu': 0.0001285076141357422, 'py_compute': 1.0500628566741943,
                            'load_model': 0.15180516242980957, 'overall_query_latency': 3.1019253730773926},
             'In-Db-opt': {'overall_query_latency': 3.363051715, 'diff': -0.0075932200000004,
                           'data_query_time_spi': 0.079331068, 'model_init_time': 0.01077707,
                           'data_query_time': 2.017169669, 'mem_allocate_time': 0.007546996,
                           'python_compute_time': 1.327511756, 'py_conver_to_tensor': 0.0031790733337402344,
                           'py_compute': 1.0584447383880615, 'py_overall_duration': 1.0742435455322266,
                           'py_diff': 0.012619733810424805}, },
    "320k": {'out-DB-cpu': {'data_query_time': 1.8355934619903564, 'py_conver_to_tensor': 0.07959046363830566,
                            'tensor_to_gpu': 0.0002346038818359375, 'py_compute': 2.173098516464233,
                            'load_model': 0.1675722599029541, 'overall_query_latency': 7.379018783569336},
             'In-Db-opt': {'data_query_time_spi': 0.150363744, 'mem_allocate_time': 0.015541999,
                           'python_compute_time': 2.647950218, 'diff': -0.01560269500000011,
                           'model_init_time': 0.010926438, 'overall_query_latency': 6.682675587,
                           'data_query_time': 4.008196236, 'py_conver_to_tensor': 0.080591365814208984,
                           'py_compute': 2.142165422439575, 'py_overall_duration': 2.181426525115967,
                           'py_diff': 0.0334019660949707},
             },
    "640k": {
        'out-DB-cpu': {'data_query_time': 3.543272018432617, 'py_conver_to_tensor': 0.03744659423828125,
                       'tensor_to_gpu': 0.0002484321594238281, 'py_compute': 6.131111097335815,
                       'load_model': 0.1593015193939209, 'overall_query_latency': 17.61525869369507},
        'In-Db-opt': {'data_query_time_spi': 0.436323265, 'data_query_time': 5.173970515,
                      'overall_query_latency': 12.380371801999999, 'diff': -0.06071819099999942,
                      'mem_allocate_time': 0.060575582, 'python_compute_time': 7.137082731,
                      'model_init_time': 0.008600365, 'py_conver_to_tensor': 0.023477554321289062,
                      'py_compute': 6.169823884963989, 'py_overall_duration': 6.283113241195679,
                      'py_diff': 0.08981180191040039},
    },

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

ax.set_ylim(top=10000)

ax.set_xticks(indices)
ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)

# ax.legend(fontsize=set_lgend_size - 2, ncol=2, )
ax.legend(fontsize=15, ncol=1, loc='upper left')

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
ax.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/exp_imgs/int_format/macro_data_scale.pdf")
fig.savefig(f"./internal/ml/model_slicing/exp_imgs/int_format/macro_data_scale_sys.pdf",
            bbox_inches='tight')
