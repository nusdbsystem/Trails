import json
import matplotlib.pyplot as plt
import numpy as np

# Assume these are the names and corresponding JSON files of your datasets
datasets_wo_cache = {
    'Frappe': {'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                      '/time_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json',
               'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                      '/time_score_mlp_sp_frappe_batch_size_32_cuda:0_express_flow.json'},

    'Diabete': {'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                       '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json',
                'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                       '/time_score_mlp_sp_uci_diabetes_batch_size_32_cuda:0_express_flow.json'},

    'Criteo': {'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                      '/time_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json',
               'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                      '/time_score_mlp_sp_criteo_batch_size_32_cuda:0_express_flow.json'},

    'c10': {'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                   '/time_score_nasbench201_cifar10_batch_size_32_cpu_express_flow.json',
            'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                   '/time_score_nasbench201_cifar10_batch_size_32_cuda:0_express_flow.json'},
    'c100': {'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                    '/time_score_nasbench201_cifar100_batch_size_32_cpu_express_flow.json',
             'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                    '/time_score_nasbench201_cifar100_batch_size_32_cuda:0_express_flow.json'},

    'IN-16': {'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                     '/time_score_nasbench201_ImageNet16-120_batch_size_32_cpu_express_flow.json',
              'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                     '/time_score_nasbench201_ImageNet16-120_batch_size_32_cuda:0_express_flow.json'},
}

datasets_embedding_cache = {
    'Frappe': {'cpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/'
                      '/time_score_mlp_sp_frappe_batch_size_32_cpu_express_flow.json',
               'gpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/'
                      '/time_score_mlp_sp_frappe_batch_size_32_cuda:0_express_flow.json'},

    'Diabete': {'cpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/'
                       '/time_score_mlp_sp_uci_diabetes_batch_size_32_cpu_express_flow.json',
                'gpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/'
                       '/time_score_mlp_sp_uci_diabetes_batch_size_32_cuda:0_express_flow.json'},

    'Criteo': {'cpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/'
                      '/time_score_mlp_sp_criteo_batch_size_32_cpu_express_flow.json',
               'gpu': './internal/ml/model_selection/exp_result_sever_filtering_cache/'
                      '/time_score_mlp_sp_criteo_batch_size_32_cuda:0_express_flow.json'},

    'c10': {'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                   '/time_score_nasbench201_cifar10_batch_size_32_cpu_express_flow.json',
            'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                   '/time_score_nasbench201_cifar10_batch_size_32_cuda:0_express_flow.json'},
    'c100': {'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                    '/time_score_nasbench201_cifar100_batch_size_32_cpu_express_flow.json',
             'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                    '/time_score_nasbench201_cifar100_batch_size_32_cuda:0_express_flow.json'},

    'IN-16': {'cpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                     '/time_score_nasbench201_ImageNet16-120_batch_size_32_cpu_express_flow.json',
              'gpu': './internal/ml/model_selection/exp_result_sever_wo_cache'
                     '/time_score_nasbench201_ImageNet16-120_batch_size_32_cuda:0_express_flow.json'},
}

# Set your plot parameters
bar_width = 0.25
opacity = 0.8
set_font_size = 20  # Set the font size
set_lgend_size = 15
set_tick_size = 12
cpu_colors = ['#729ECE', '#FFB579', '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#8E44AD', '#C0392B']
gpu_colors = ['#98DF8A', '#D62728', '#1ABC9C', '#9B59B6', '#34495E', '#16A085', '#27AE60', '#2980B9']
hatches = ['/', '\\', 'x', 'o', 'O', '.', '*', '//', '\\\\', 'xx', 'oo', 'OO', '..', '**']
# hatches = ['', '', '', '', '']

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from matplotlib.ticker import FuncFormatter
# # This is your custom formatter function
# def thousands_format(x, pos):
#     return f'{x * 1e-3}K'
# # This creates a formatter using your function
# formatter = FuncFormatter(thousands_format)

for img_id, datasets in enumerate([datasets_wo_cache, datasets_embedding_cache]):
    # Define the grid for subplots with widths ratio as 1:2
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

    fig = plt.figure(figsize=(6.4, 4.5))
    fig.subplots_adjust(wspace=0.01)  # adjust the space between

    # Split your datasets into two groups
    if img_id == 0:
        # here the no cahced one
        datasets_left = dict(list(datasets.items())[:2])
        datasets_right = dict(list(datasets.items())[2:])
        name = "wo_cache"
    else:
        # here the cahced one
        datasets_left = dict(list(datasets.items())[:3])
        datasets_right = dict(list(datasets.items())[3:])
        name = "with_cache"

    for idx, datasets in enumerate([datasets_left, datasets_right]):
        # Create a subplot with custom width
        ax = plt.subplot(gs[idx])

        for i, (dataset_name, json_files) in enumerate(datasets.items()):
            # Load the JSON data for cpu
            with open(json_files['cpu']) as f:
                data_cpu = json.load(f)

            # Load the JSON data for gpu
            with open(json_files['gpu']) as f:
                data_gpu = json.load(f)

            # Plot bars for CPU
            bottom_init = sum(data_cpu['track_io_model_init'][2:])
            bottom_load_release = bottom_init

            ax.bar(i - bar_width / 2, sum(data_cpu['track_io_model_init'][2:]), bar_width,
                   alpha=opacity, color=cpu_colors[0], hatch=hatches[0], edgecolor='black',
                   label='(CPU) M Init' if i == 0 else "")

            ax.bar(i - bar_width / 2, data_cpu['compute_latency'], bar_width,
                   alpha=opacity, color=cpu_colors[1], hatch=hatches[1], edgecolor='black',
                   label='(CPU) TFMEM' if i == 0 else "",
                   bottom=bottom_load_release)

            # Plot bars for GPU
            bottom_init_gpu = sum(data_gpu['track_io_model_init'][2:])
            bottom_load_release_gpu = bottom_init_gpu + sum(data_gpu['track_io_model_load'][2:])

            ax.bar(i + bar_width / 2, sum(data_gpu['track_io_model_init'][2:]), bar_width,
                   alpha=opacity, color=gpu_colors[0], hatch=hatches[0], edgecolor='black',
                   label='(GPU) M Init' if i == 0 else "")

            ax.bar(i + bar_width / 2,
                   sum(data_gpu['track_io_model_load'][2:]),
                   bar_width,
                   alpha=opacity, color=gpu_colors[1], hatch=hatches[1], edgecolor='black',
                   label='(GPU) M Load' if i == 0 else "",
                   bottom=bottom_init_gpu)

            ax.bar(i + bar_width / 2, data_gpu['compute_latency'], bar_width,
                   alpha=opacity, color=gpu_colors[2], hatch=hatches[2], edgecolor='black',
                   label='(GPU) TFMEM' if i == 0 else "",
                   bottom=bottom_load_release_gpu)

            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(datasets.keys(), fontsize=set_font_size)

        # Set axis labels and legend for first subplot only
        if idx == 0:
            ax.set_ylabel('Latency (s)', fontsize=set_font_size)
            ax.legend().remove()  # remove the legend
        else:
            # ax.yaxis.set_major_formatter(formatter)
            ax.set_ylim(0, 1800)
            ax.legend(fontsize=set_lgend_size, loc='upper right', ncol=1)

        # increase x size
        ax.tick_params(axis='x', which='major', labelsize=set_tick_size + 5)

    fig.tight_layout()
    # plt.show()
    print(f"saving to ./internal/ml/model_selection/exp_result/filter_latency_{name}.pdf")
    fig.savefig(f"./internal/ml/model_selection/exp_result/filter_latency_{name}.pdf",
                bbox_inches='tight')
