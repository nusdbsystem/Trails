from src.tools.io_tools import read_json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter


def thousands_formatter(x, pos):
    if x >= 1e3:
        return '{:.1f}k'.format(x * 1e-3)
    else:
        return '{:.1f}'.format(x)


thousands_format = FuncFormatter(thousands_formatter)

# Set your plot parameters
bar_width = 0.25
linewidth=2.5
opacity = 0.8
set_font_size = 18  # Set the font size
set_lgend_size = 11
set_tick_size = 18
cpu_colors = ['#729ECE', '#FFB579', '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#8E44AD', '#C0392B']
gpu_colors = ['#98DF8A', '#D62728', '#1ABC9C', '#9B59B6', '#34495E', '#16A085', '#27AE60', '#2980B9']
hatches = ['/', '\\', 'x', 'o', 'O', '.', '*', '//', '\\\\', 'xx', 'oo', 'OO', '..', '**']

# Assume these are the names and corresponding JSON files of your datasets
datasets_wo_cache = {
    'Frappe': {
        'with cache': './internal/ml/model_selection/exp_result_sever_filtering_cache'
                      '/resource_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json',
        'w/o cache': './internal/ml/model_selection/exp_result_sever_wo_cache'
                     '/resource_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json'},

    'Diabetes': {
        'with cache': './internal/ml/model_selection/exp_result_sever_filtering_cache'
                      '/resource_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json',
        'w/o cache': './internal/ml/model_selection/exp_result_sever_wo_cache'
                     '/resource_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json'},

    'Criteo': {
        'with cache': './internal/ml/model_selection/exp_result_sever_filtering_cache'
                      '/resource_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json',
        'w/o cache': './internal/ml/model_selection/exp_result_sever_wo_cache'
                     '/resource_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json'}
}


def plot_memory_usage(params, interval=0.5):
    # Set up the gridspec layout
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    fig = plt.figure(figsize=(6.4, 4.5))
    # Adjust the space between
    fig.subplots_adjust(hspace=0.25)

    line_idx = 0
    # 1. first plot
    ax_frappe_uci = fig.add_subplot(gs[0])
    for dataset_name, value in params.items():
        if dataset_name == "Criteo":
            continue
        for is_cache, file_path in value.items():
            metrics = read_json(file_path)
            # Extract GPU memory usage for device 0
            memory_usage_lst = metrics['memory_usage']
            # Create a time list
            times = [interval * i for i in range(len(memory_usage_lst))]

            if is_cache == "with cache":
                linestyle = "--"
            else:
                linestyle = "-"
            ax_frappe_uci.plot(times, memory_usage_lst, label=f"{dataset_name} ({is_cache})",
                               linestyle=linestyle, linewidth=linewidth)

    ax_frappe_uci.set_ylabel('Memory (MB)', fontsize=set_font_size)
    ax_frappe_uci.legend(fontsize=set_lgend_size)
    ax_frappe_uci.set_xticklabels([])  # Hide the x-axis labels for the top plot
    ax_frappe_uci.tick_params(axis='both', which='major', labelsize=set_tick_size)
    ax_frappe_uci.set_xscale("symlog")
    ax_frappe_uci.grid(True)
    # ax_frappe_uci.set_ylim(10, None)
    ax_frappe_uci.yaxis.set_major_formatter(thousands_format)

    # 2. second plot
    ax_criteo = fig.add_subplot(gs[1], sharex=ax_frappe_uci)
    for dataset_name, value in params.items():
        if dataset_name == "Criteo":
            for is_cache, file_path in value.items():
                metrics = read_json(file_path)
                # Extract GPU memory usage for device 0
                memory_usage_lst = metrics['memory_usage']
                # Create a time list
                times = [interval * i for i in range(len(memory_usage_lst))]
                if is_cache == "with cache":
                    linestyle = "--"
                else:
                    linestyle = "-"
                ax_criteo.plot(times, memory_usage_lst, label=f"{dataset_name} ({is_cache})",
                               linestyle=linestyle, linewidth=linewidth)

    ax_criteo.legend(fontsize=set_lgend_size)
    ax_criteo.set_ylabel('Memory (MB)', fontsize=set_font_size)
    ax_criteo.set_xlabel('Time (Seconds)', fontsize=set_font_size)
    ax_criteo.tick_params(axis='both', which='major', labelsize=set_tick_size)
    ax_criteo.set_xscale("symlog")
    ax_criteo.grid(True)
    # ax_criteo.set_ylim(10, None)
    ax_criteo.yaxis.set_major_formatter(thousands_format)

    # plt.show()
    print(f"saving to ./internal/ml/model_selection/exp_result/filter_latency_memory_cache.pdf")
    fig.savefig(f"./internal/ml/model_selection/exp_result/filter_latency_memory_cache.pdf",
                bbox_inches='tight')


# Call the function
plot_memory_usage(datasets_wo_cache)
