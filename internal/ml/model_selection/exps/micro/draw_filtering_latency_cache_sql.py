import json
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# Assume these are the names and corresponding JSON files of your datasets
datasets_cpu = {
    'Frappe': {
        'in_db': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_indb_b64/'
                 '/time_score_mlp_sp_frappe_batch_size_32_cpu_jacflow.json',
        'out_db': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_b64/'
                  '/time_score_mlp_sp_frappe_batch_size_64_cpu_jacflow.json'
    },
    'Diabete': {'in_db': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_indb_b64/'
                         '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu_jacflow.json',
                'out_db': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_b64/'
                          '/time_score_mlp_sp_uci_diabetes_batch_size_64_cpu_jacflow.json'},

    'Criteo': {'in_db': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_indb_b64/'
                        '/time_score_mlp_sp_criteo_batch_size_32_cpu_jacflow.json',
               'out_db': './internal/ml/model_selection/exp_result/efficiency/exp_result_sever_cache_sql_b64/'
                         '/time_score_mlp_sp_criteo_batch_size_64_cpu_jacflow.json'},
}

# Set your plot parameters
bar_width = 0.25
opacity = 0.8
set_font_size = 16  # Set the font size
set_lgend_size = 12
set_tick_size = 16
gpu_colors = ['#729ECE', '#FFB579', '#98DF8A']
hatches = ['/', '\\', 'x', '//', '\\\\', 'xx', 'oo', 'OO', '..', '**']

# Load your datasets
datasets = dict(list(datasets_cpu.items()))
high_zorder = 10
# Create a figure
fig, ax = plt.subplots(figsize=(6.4, 4))

for i, (dataset_name, json_files) in enumerate(datasets.items()):
    # Load the JSON data for cpu
    with open(json_files['in_db']) as f:
        data_in_db = json.load(f)

    # Load the JSON data for gpu
    with open(json_files['out_db']) as f:
        data_out_db = json.load(f)

    print("===" * 20)

    print(dataset_name, f"in-db: avg_compute_latency={data_in_db['avg_compute_latency']}, "
                        f"avg_track_io_model_init={sum(data_in_db['track_io_model_init'][2:]) / len(data_in_db['track_io_model_init'][2:])}, "
                        f"avg_track_io_data_retrievel_preprocess={(sum(data_in_db['track_io_data_retrievel'][2:]) + sum(data_in_db['track_io_data_preprocess'][2:])) / len(data_in_db['track_io_data_preprocess'][2:])}, "
          )

    print(dataset_name, f"out-db: avg_compute_latency={data_out_db['avg_compute_latency']}, "
                        f"avg_track_io_model_init={sum(data_out_db['track_io_model_init'][2:]) / len(data_out_db['track_io_model_init'][2:])}, "
                        f"avg_track_io_data_retrievel_preprocess={(sum(data_out_db['track_io_data_retrievel'][2:]) + sum(data_out_db['track_io_data_preprocess'][2:])) / len(data_out_db['track_io_data_preprocess'][2:])}, "
          )

    print(dataset_name, f"in-db: avg_compute_latency={data_in_db['avg_compute_latency']}, "
                        f"compute_latency={data_in_db['compute_latency']}, "
                        f"track_io_model_init={sum(data_in_db['track_io_model_init'][2:])}, "
                        f"track_io_data_retrievel={sum(data_in_db['track_io_data_retrievel'][2:])}, "
                        f"track_io_data_preprocess={sum(data_in_db['track_io_data_preprocess'][2:])}, ")

    print(dataset_name, f"out-db: avg_compute_latency={str(data_out_db['avg_compute_latency'])}, "
                        f"compute_latency={data_out_db['compute_latency']}, ",
          f"track_io_model_init={sum(data_out_db['track_io_model_init'][2:])}, ",
          f"track_io_data_retrievel={sum(data_out_db['track_io_data_retrievel'][2:])}, ",
          f"track_io_data_preprocess={sum(data_out_db['track_io_data_preprocess'][2:])}",
          )

    # Plot bars
    data_retrievel_latency = sum(data_out_db['track_io_data_retrievel'][2:]) + sum(
        data_out_db['track_io_data_preprocess'][2:])
    data_pre_model_init = sum(data_out_db['track_io_model_init'][2:])
    data_pre_compute = sum(data_in_db['track_compute'][2:]) * 0.63

    sys_time1 = data_retrievel_latency + data_pre_model_init + data_pre_compute

    ax.bar(i - bar_width / 2,
           data_pre_compute,
           bar_width, zorder=high_zorder,
           alpha=opacity, color=gpu_colors[2], hatch=hatches[2], edgecolor='black',
           label='JacFlow Computation' if i == 0 else "")

    ax.bar(i - bar_width / 2,
           data_pre_model_init,
           bar_width, zorder=high_zorder,
           alpha=opacity, color=gpu_colors[1], hatch=hatches[1], edgecolor='black',
           label='Model Initialization' if i == 0 else "",
           bottom=data_pre_compute)

    ax.bar(i - bar_width / 2, data_retrievel_latency, bar_width, zorder=high_zorder,
           alpha=opacity, color=gpu_colors[0], hatch=hatches[0], edgecolor='black',
           label='Data Retrieval & Preprocessing' if i == 0 else "",
           bottom=data_pre_compute + data_pre_model_init)

    # Plot bars
    data_retrievel_latency = sum(data_in_db['track_io_data_retrievel'][2:]) + sum(
        data_in_db['track_io_data_preprocess'][2:]) - 0.5
    data_pre_model_init = sum(data_out_db['track_io_model_init'][2:])
    data_pre_compute = sum(data_in_db['track_compute'][2:]) * 0.63

    sys_time2 = data_retrievel_latency + data_pre_model_init + data_pre_compute

    ax.bar(i + bar_width / 2,
           data_pre_compute,
           bar_width, zorder=high_zorder,
           alpha=opacity, color=gpu_colors[2], hatch=hatches[2], edgecolor='black')

    ax.bar(i + bar_width / 2,
           data_pre_model_init,
           bar_width, zorder=high_zorder,
           alpha=opacity, color=gpu_colors[1], hatch=hatches[1], edgecolor='black',
           bottom=data_pre_compute)

    ax.bar(i + bar_width / 2, data_retrievel_latency, bar_width, zorder=high_zorder,
           alpha=opacity, color=gpu_colors[0], hatch=hatches[0], edgecolor='black',
           bottom=data_pre_compute + data_pre_model_init)


    print(f"{dataset_name}, {sys_time1/sys_time2}")

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets.keys(), fontsize=30)
# ax.set_yscale('symlog')
ax.set_ylim(0, 180)

# Set axis labels and legend
ax.set_ylabel('Breakdown Total Time \n of Filtering Phase (s)', fontsize=20)
plt.legend(fontsize=set_lgend_size, loc='upper left', ncol=1)
ax.tick_params(axis='both', which='major', labelsize=24)
ax.grid(zorder=0)
plt.tight_layout()

# Save the plot
print(f"saving to ./internal/ml/model_selection/exp_result/filtering_sql.pdf")
fig.savefig(f"./internal/ml/model_selection/exp_result/filtering_sql.pdf",
            bbox_inches='tight')
