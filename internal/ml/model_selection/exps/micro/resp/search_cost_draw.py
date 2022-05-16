from typing import List

from src.tools.compute import sample_in_log_scale_new
from src.tools.io_tools import read_json
from exps.draw_tab_lib import draw_structure_data_anytime
import numpy as np
from pprint import pprint
from exps.micro.resp.benchmark_combinations import export_warm_up_move_proposal, filter_refinment_fully_training
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_dataset_parameters(dataset):
    parameters = {
        "frappe": {
            "epoch": 19,
            "train_re": "./internal/ml/model_selection/exp_result/train_base_line_re_frappe_epoch_19.json",
            "train_rs": "./internal/ml/model_selection/exp_result/train_base_line_rs_frappe_epoch_19.json",
            "train_rl": "./internal/ml/model_selection/exp_result/train_base_line_rl_frappe_epoch_19.json",
            "express_flow": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_express_flow.json",
            "express_flow_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_express_flow_p1.json",
            "fisher": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_fisher.json",
            "fisher_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_fisher_p1.json",
            "grad_norm": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_grad_norm.json",
            "grad_norm_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_grad_norm_p1.json",
            "grasp": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_grasp.json",
            "grasp_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_grasp_p1.json",
            "nas_wot": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_nas_wot.json",
            "nas_wot_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_nas_wot_p1.json",
            "ntk_cond_num": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_cond_num.json",
            "ntk_cond_num_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_cond_num_p1.json",
            "ntk_trace": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_trace.json",
            "ntk_trace_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_trace_p1.json",
            "ntk_trace_approx": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_trace_approx.json",
            "ntk_trace_approx_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_ntk_trace_approx_p1.json",
            "snip": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_snip.json",
            "snip_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_snip_p1.json",
            "synflow": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_synflow.json",
            "synflow_p1": "./internal/ml/model_selection/exp_result/res_end_2_end_mlp_sp_frappe_-1_10_synflow_p1.json",
            "mx_value": 98.14,
            "y_lim": [97.5, 98.08],
            "figure_size": (6.2, 4.71),
            "datasetfg_name": dataset,
            "annotations": [],  # ["TabNAS", 97.68, 324.8/60],
            "remove_n_points": 2,
        },
    }

    return parameters.get(dataset, None)


def sample_some_points(x_array, y_2d_array, save_points, remove_n_points=1) -> (List, List):
    result_x_array = []
    result_y_array = []
    for run_id, time_list in enumerate(x_array):
        indices = sample_in_log_scale_new(time_list, save_points)
        each_run_x_array = [time_list[i] for i in indices]
        each_run_y_array = [y_2d_array[run_id][int(i)] for i in indices]

        result_x_array.append(each_run_x_array[:-remove_n_points] if remove_n_points != 0 else each_run_x_array)
        result_y_array.append(each_run_y_array[:-remove_n_points] if remove_n_points != 0 else each_run_y_array)

    return result_x_array, result_y_array


def find_time_for_target_accuracy(elements, target_accuracy_index, target_algo='express_flow'):
    target_accuracy = None
    results = {}

    # Find target accuracy for the algorithm
    for e in elements:
        x, y, algo = e
        x = [np.median(val) for val in zip(*x)]  # Calculate median per column
        y = [np.median(val) for val in zip(*y)]  # Calculate median per column
        if algo == target_algo:
            if target_accuracy_index < len(y):
                target_accuracy = y[target_accuracy_index]
            else:
                print(f"{target_algo} does not have a {target_accuracy_index}-th accuracy.")
                return
            break

    # Interpolate for other algorithms
    for e in elements:
        x, y, algo = e
        x = [np.median(val) for val in zip(*x)]  # Calculate median per column
        y = [np.median(val) for val in zip(*y)]  # Calculate median per column
        if target_accuracy > max(y):
            results[algo.split(" - ")[0]] = '-'
        else:
            if target_accuracy in y:
                estimated_time = x[y.index(target_accuracy)]
            else:
                estimated_time = np.interp(target_accuracy, y, x)
            results[algo.split(" - ")[0]] = estimated_time * 60
    pprint(f"Achieving target AUC = {target_accuracy}, time usage in seconds: ")
    pprint(results)


def generate_and_draw_data(dataset):
    params = get_dataset_parameters(dataset)
    result_dir = "./internal/ml/model_selection/exp_result/"
    json_keys = [k for k, v in params.items() if isinstance(v, str) and v.endswith('.json')]

    # 1. compare training-free and two-phase under various tfmem
    all_lines = []
    for key in json_keys:
        result = read_json(params[key])

        # make it as 5 run
        if isinstance(result["sys_time_budget"][0], float):
            x_array = [result["sys_time_budget"] for _ in result["sys_acc"]]
        else:
            x_array = result["sys_time_budget"]
        sampled_x, sampled_y = sample_some_points(
            x_array=x_array,
            y_2d_array=result["sys_acc"],
            save_points=7,
            remove_n_points=0)

        all_lines.append([sampled_x, sampled_y, key])

    # 3. draw the figure for traing-free/train-based and two phase
    draw_structure_data_anytime(
        all_lines=all_lines,
        dataset=params['datasetfg_name'],
        name_img=f"{result_dir}/anytime_{dataset}",
        max_value=params['mx_value'],
        figure_size=params['figure_size'],
        annotations=params['annotations'],
        y_ticks=params['y_lim'],
        x_ticks=[0.01, None]
    )

    # 2. draw filtering phase only
    selected_lines = []
    for line in all_lines:
        if line[-1].split("_")[-1] == "p1":
            selected_lines.append(line)

    find_time_for_target_accuracy(elements=selected_lines, target_accuracy_index=1, target_algo='express_flow_p1')
    draw_structure_data_anytime(
        all_lines=selected_lines,
        dataset=params['datasetfg_name'],
        name_img=f"{result_dir}/anytime_p1_only_{dataset}",
        max_value=params['mx_value'],
        figure_size=params['figure_size'],
        annotations=params['annotations'],
        y_ticks=params['y_lim'],
        x_ticks=[0.01, 100])

    # 3. training-free, training-based, warm-up and move-proposal with (snip, naswot, synflow, expressFlow)
    print('Computing EA+warm up(3k), and EA + Move ')
    for tfmem in ["synflow", "nas_wot", "snip", "express_flow"]:
        ea_warm_up, ea_move = export_warm_up_move_proposal(tfmem)
        all_lines.append(ea_warm_up)
        all_lines.append(ea_move)
    filter_full_train = filter_refinment_fully_training()
    all_lines.append(filter_full_train)

    find_time_for_target_accuracy(elements=all_lines, target_accuracy_index=3, target_algo='express_flow')

    # 3. draw graph for table 4
    selected_lines = []
    for line in all_lines:
        if line[-1] in [
            "synflow_p1", "nas_wot_p1", "snip_p1",
            "EA + warmup-synflow (3K)", "EA + warmup-nas_wot (3K)", "EA + warmup-snip (3K)",
            "EA + warmup-express_flow (3K)",
            "EA + move-synflow", "EA + move-nas_wot", "EA + move-snip", "EA + move-express_flow",
            "express_flow", "Filtering + Refinement (Fully Train)", "train_re", "train_rs", "train_rl"]:
            selected_lines.append(line)

    draw_structure_data_anytime(
        all_lines=selected_lines,
        dataset=params['datasetfg_name'],
        name_img=f"{result_dir}/anytime_combines_{dataset}",
        max_value=params['mx_value'],
        figure_size=params['figure_size'],
        annotations=params['annotations'],
        y_ticks=params['y_lim'],
        x_ticks=[0.01, 100]
    )


# Choose dataset to process
dataset = "frappe"
generate_and_draw_data(dataset)
