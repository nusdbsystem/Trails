from src.tools.io_tools import read_json
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from exps.draw_tab_lib import export_legend

# Set your plot parameters
bar_width = 0.25
linewidth=2.5
opacity = 0.8
set_font_size = 15  # Set the font size
set_lgend_size = 15
set_tick_size = 15
cpu_colors = ['#729ECE', '#FFB579', '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#8E44AD', '#C0392B']
gpu_colors = ['#98DF8A', '#D62728', '#1ABC9C', '#9B59B6', '#34495E', '#16A085', '#27AE60', '#2980B9']
hatches = ['/', '\\', 'x', 'o', 'O', '.', '*', '//', '\\\\', 'xx', 'oo', 'OO', '..', '**']
# hatches = ['', '', '', '', '']
line_styles = ['-', '--', '-.', (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1))]


# Assume these are the names and corresponding JSON files of your datasets
datasets_wo_cache = {
    'frappe': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_frappe_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json'
    },

    'diabetes': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_uci_diabetes_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json'
    },

    'criteo': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_criteo_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json'
    },

    'c10': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_cifar10_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_cifar10_batch_size_32_cpu_express_flow.json'
    },

    'c100': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_cifar100_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_cifar100_batch_size_32_cpu_express_flow.json'
    },

    'IN-16': {
        'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_ImageNet16-120_batch_size_32_cuda:0_express_flow.json',
        'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
               '/resource_score_nasbench201_ImageNet16-120_batch_size_32_cpu_express_flow.json'
    },
}


def plot_memory_usage(params, interval=0.5):
    # Set up the gridspec layout
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    fig = plt.figure(figsize=(6.4, 4.5))
    # Adjust the space between
    fig.subplots_adjust(hspace=0.25)

    # 1. first plot
    ax_gpu = fig.add_subplot(gs[0])
    for dataset_name, value in params.items():
        metrics = read_json(params[dataset_name]["gpu"])
        # Extract GPU memory usage for device 0
        gpu_mem_device_0 = [mem[2] for mem in metrics['gpu_usage'] if mem[0] == 0]
        # count from the 1st non-zero or near zero position
        break_point = 0
        for idx, val in enumerate(gpu_mem_device_0[:-1]):
            if gpu_mem_device_0[idx + 1] > 200:
                break_point = idx
                break
        gpu_mem_device_0 = gpu_mem_device_0[break_point:]
        mem_host = metrics['memory_usage'][break_point:]
        total_memory_usage = [a + b for a, b in zip(gpu_mem_device_0, mem_host)]

        # Create a time list
        times = [interval * i for i in range(len(total_memory_usage))]
        ax_gpu.plot(times, total_memory_usage, label=dataset_name, linestyle=line_styles[idx % len(line_styles)], linewidth=linewidth)
    ax_gpu.set_ylabel('Memory (MB)', fontsize=set_font_size)
    ax_gpu.legend()
    ax_gpu.set_xticklabels([])  # Hide the x-axis labels for the top plot
    ax_gpu.tick_params(axis='both', which='major', labelsize=set_tick_size)
    ax_gpu.set_xscale("symlog")
    ax_gpu.set_yscale("symlog")
    ax_gpu.grid(True)
    ax_gpu.set_ylim(100, 10000)

    # 2. second plot
    ax_cpu = fig.add_subplot(gs[1], sharex=ax_gpu)
    for dataset_name, value in params.items():
        metrics = read_json(params[dataset_name]["cpu"])
        # Extract GPU memory usage for device 0
        memory_usage = metrics['memory_usage']
        # Create a time list
        times = [interval * i for i in range(len(memory_usage))]
        ax_cpu.plot(times, memory_usage, label=dataset_name, linestyle=line_styles[idx % len(line_styles)], linewidth=linewidth)

    ax_cpu.set_ylabel('Memory (MB)', fontsize=set_font_size)
    ax_cpu.set_xlabel('Time (Seconds)', fontsize=set_font_size)
    ax_cpu.legend()
    # Setting features for ax2
    ax_cpu.tick_params(axis='both', which='major', labelsize=set_tick_size)
    ax_cpu.set_xscale("symlog")
    ax_cpu.set_yscale("symlog")
    ax_cpu.grid(True)
    # ax_cpu.set_ylim(10, None)

    # global setting
    export_legend(fig)

    print(f"saving to ./internal/ml/model_selection/exp_result/filter_latency_memory.pdf")
    fig.savefig(f"./internal/ml/model_selection/exp_result/filter_latency_memory.pdf",
                bbox_inches='tight')


# Call the function
plot_memory_usage(datasets_wo_cache)
