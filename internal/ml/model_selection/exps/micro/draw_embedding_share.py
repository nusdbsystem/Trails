import json
import matplotlib.pyplot as plt
import numpy as np

# Assume these are the names and corresponding JSON files of your datasets
datasets_cpu = {
    'Frappe': {'cache': './internal/ml/model_selection/exp_result_sever_filtering_cache/'
                        '/time_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json',
               'no_cache': './internal/ml/model_selection/exp_result_sever/'
                           '/time_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json'},

    'Diabete': {'cache': './internal/ml/model_selection/exp_result_sever_filtering_cache/'
                         '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json',
                'no_cache': './internal/ml/model_selection/exp_result_sever/'
                            '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json'},

    'Criteo': {'cache': './internal/ml/model_selection/exp_result_sever_filtering_cache/'
                        '/time_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json',

               'no_cache': './internal/ml/model_selection/exp_result_sever/'
                           '/time_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json'
               },

}

# Set your plot parameters
bar_width = 0.25
opacity = 0.8
set_font_size = 20  # Set the font size
set_lgend_size = 15
set_tick_size = 20
cpu_colors = ['#729ECE', '#FFB579', '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#8E44AD', '#C0392B']
gpu_colors = ['#98DF8A', '#D62728', '#1ABC9C', '#9B59B6', '#34495E', '#16A085', '#27AE60', '#2980B9']
hatches = ['/', '\\', 'x', 'o', 'O', '.', '*', '//', '\\\\', 'xx', 'oo', 'OO', '..', '**']

# Load your datasets
datasets = dict(list(datasets_cpu.items()))

# Create a figure
fig, ax = plt.subplots(figsize=(6.4, 4.5))

for i, (dataset_name, json_files) in enumerate(datasets.items()):
    # Load the JSON data for cpu
    with open(json_files['cache']) as f:
        data_cache = json.load(f)

    # Load the JSON data for gpu
    with open(json_files['no_cache']) as f:
        data_non_cache = json.load(f)

    # Plot bars for cpu
    ax.bar(i - bar_width / 2, data_cache['io_latency'], bar_width,
           alpha=opacity, color=cpu_colors[0], hatch=hatches[0], edgecolor='black',
           label='(Emb. Cache) Model Init' if i == 0 else "")

    ax.bar(i - bar_width / 2, data_cache['compute_latency'], bar_width,
           alpha=opacity, color=cpu_colors[1], hatch=hatches[1], edgecolor='black',
           label='(Emb. Cache) TFMEM' if i == 0 else "",
           bottom=data_cache['io_latency'])

    # Plot bars for gpu
    ax.bar(i + bar_width / 2, data_non_cache['io_latency'], bar_width,
           alpha=opacity, color=gpu_colors[0], hatch=hatches[2], edgecolor='black',
           label='Model Init' if i == 0 else "")

    ax.bar(i + bar_width / 2, data_non_cache['compute_latency'], bar_width,
           alpha=opacity, color=gpu_colors[2], hatch=hatches[2], edgecolor='black',
           label='TFMEM' if i == 0 else "",
           bottom=data_non_cache['io_latency'])

ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets.keys(), fontsize=set_font_size)
ax.set_yscale('symlog')
ax.set_ylim(0, 3000)

# Set axis labels and legend
ax.set_ylabel('Latency (s)', fontsize=set_font_size)
ax.legend(fontsize=set_lgend_size, loc='upper left', ncol=1)
ax.tick_params(axis='both', which='major', labelsize=set_tick_size)
plt.tight_layout()

# Save the plot
fig.savefig(f"./internal/ml/model_selection/exp_result_sever/embedding_cache.pdf",
            bbox_inches='tight')
