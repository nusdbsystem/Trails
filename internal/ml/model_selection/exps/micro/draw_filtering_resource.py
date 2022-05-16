import matplotlib.pyplot as plt
from src.tools.io_tools import read_json
import seaborn as sns


def plot_resource_usage(files, labels):
    # Initialize plots for CPU, Memory and GPU usage
    fig_cpu, ax_cpu = plt.subplots()
    fig_mem, ax_mem = plt.subplots()
    fig_gpu, ax_gpu = plt.subplots()

    # Define a color palette
    colors = sns.color_palette("hsv", len(files))

    # Define line styles for different systems
    linestyles = ['-', '--']

    # Loop over each pair of files
    for idx, (file_pair, label) in enumerate(zip(files, labels)):
        # Load data from json files
        data1 = read_json(file_pair[0])  # system1
        data2 = read_json(file_pair[1])  # system2

        # Only take the first GPU usage data
        data1['gpu_usage'] = [data[2] for i, data in enumerate(data1['gpu_usage']) if data[0] == 0]
        data2['gpu_usage'] = [data[2] for i, data in enumerate(data2['gpu_usage']) if data[0] == 0]

        # Define a color for this dataset
        color = colors[idx]

        # Plot CPU usage
        ax_cpu.plot(data1['cpu_usage'], label=f'{label} System 1 CPU', linestyle=linestyles[0], color=color)
        ax_cpu.plot(data2['cpu_usage'], label=f'{label} System 2 CPU', linestyle=linestyles[1], color=color)

        # Plot Memory usage
        ax_mem.plot(data1['memory_usage'], label=f'{label} System 1 Memory', linestyle=linestyles[0], color=color)
        ax_mem.plot(data2['memory_usage'], label=f'{label} System 2 Memory', linestyle=linestyles[1], color=color)

        # Plot GPU usage
        ax_gpu.plot(data1['gpu_usage'], label=f'{label} System 1 GPU', linestyle=linestyles[0], color=color)
        ax_gpu.plot(data2['gpu_usage'], label=f'{label} System 2 GPU', linestyle=linestyles[1], color=color)

    # Set labels and legends for each plot
    for fig, ax, usage in zip([fig_cpu, fig_mem, fig_gpu],
                              [ax_cpu, ax_mem, ax_gpu],
                              ['CPU Usage (%)', 'Memory Usage (MB)', 'GPU Memory Usage (MB)']):
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(usage)
        ax.legend(loc='upper right')

        # Set x and y axes to logarithmic scale
        ax.set_xscale('log')
        ax.set_yscale('log')

        fig.show()
# Define file pairs and labels for each pair
files = [
    ('./internal/ml/model_selection/exp_result_sever/exp_result/' \
        'resource_score_mlp_sp_frappe_batch_size_32_cuda:0.json',
     './internal/ml/model_selection/exp_result_sever/exp_result/' \
        'resource_score_mlp_sp_frappe_batch_size_32_cpu.json'),

    ('./internal/ml/model_selection/exp_result_sever/exp_result/' \
     'resource_score_mlp_sp_criteo_batch_size_32_cuda:0.json',
     './internal/ml/model_selection/exp_result_sever/exp_result/' \
     'resource_score_mlp_sp_criteo_batch_size_32_cpu.json'),

    # Add more file pairs as needed
]
labels = [
    'Frappe',
    'Criteo',
    # Add more labels as needed
]

plot_resource_usage(files, labels)

