from typing import List

from src.tools.compute import sample_in_log_scale_new
from src.tools.io_tools import read_json
from exps.draw_tab_lib import draw_structure_data_anytime
from exps.shared_args import parse_arguments

args = parse_arguments()


def sample_some_points(x_array, y_2d_array, save_points, remove_n_points=1) -> (List, List):
    result_x_array = []
    result_y_array = []
    for run_id, time_list in enumerate(x_array):
        indices = sample_in_log_scale_new(time_list, save_points)
        # Sample the list using the calculated indices
        each_run_x_array = [time_list[i] for i in indices]
        each_run_y_array = [y_2d_array[run_id][int(i)] for i in indices]

        if remove_n_points != 0:
            result_x_array.append(each_run_x_array[:-remove_n_points])
            result_y_array.append(each_run_y_array[:-remove_n_points])
        else:
            result_x_array.append(each_run_x_array)
            result_y_array.append(each_run_y_array)

    return result_x_array, result_y_array


# 'frappe, criteo, uci_diabetes'
# dataset = "frappe"
dataset = "criteo"
# dataset = "uci_diabetes"

if dataset == "uci_diabetes":
    epoch = 0
    mx_value = 67.4
    y_lim = [60, None]
    x_lim = [1.2, 1600]
    figure_size = (6.4, 3.8)
    datasetfg_name = "Diabetes"
    remove_n_points = 1
    annotations = []
elif dataset == "frappe":
    epoch = 13
    mx_value = 98.122
    y_lim = [97.6, None]
    x_lim = [2.1, 2000]
    figure_size = (6.4, 4)
    datasetfg_name = dataset
    remove_n_points = 1
    annotations = []

elif dataset == "criteo":
    epoch = 9
    mx_value = 80.337
    y_lim = [80.24, 80.345]
    x_lim = [2.1, 1600]
    figure_size = (6.4, 4)
    datasetfg_name = dataset
    annotations = []
    remove_n_points = 1
else:
    pass

train_based_res_ea = read_json(f"./internal/ml/model_selection/exp_result/train_base_line_re_{dataset}_epoch_{epoch}.json")
sampled_ea_x, sampled_ea_y = sample_some_points(x_array=[list(range(1, len(train_based_res_ea["sys_time_budget"][0]))) for ele in train_based_res_ea["sys_time_budget"]],
                                                y_2d_array=train_based_res_ea["sys_acc"],
                                                save_points=9,
                                                remove_n_points=remove_n_points)

train_based_res_rl = read_json(f"./internal/ml/model_selection/exp_result/train_base_line_rl_{dataset}_epoch_{epoch}.json")
sampled_el_x, sampled_el_y = sample_some_points(x_array=[list(range(1, len(train_based_res_rl["sys_time_budget"][0]))) for ele in train_based_res_rl["sys_time_budget"]],
                                                y_2d_array=train_based_res_rl["sys_acc"],
                                                save_points=9,
                                                remove_n_points=remove_n_points)

train_based_res_rs = read_json(f"./internal/ml/model_selection/exp_result/train_base_line_rs_{dataset}_epoch_{epoch}.json")
sampled_rs_x, sampled_rs_y = sample_some_points(x_array=[list(range(1, len(train_based_res_rs["sys_time_budget"][0]))) for ele in train_based_res_rs["sys_time_budget"]],
                                                y_2d_array=train_based_res_rs["sys_acc"],
                                                save_points=9,
                                                remove_n_points=remove_n_points)

train_based_res_rs = read_json(f"./internal/ml/model_selection/exp_result/train_base_line_bohb_{dataset}_epoch_{epoch}.json")
sampled_bohb_x, sampled_bohb_y = sample_some_points(x_array=[list(range(1, len(train_based_res_rs["sys_time_budget"][0]))) for ele in train_based_res_rs["sys_time_budget"]],
                                                    y_2d_array=train_based_res_rs["sys_acc"],
                                                    save_points=9,
                                                    remove_n_points=remove_n_points)

all_lines = [
    [sampled_ea_x, sampled_ea_y, "RE"],
    [sampled_el_x, sampled_el_y, "RL"],
    [sampled_rs_x, sampled_rs_y, "RS"],
    [sampled_bohb_x, sampled_bohb_y, "BOHB"]
]

print(f"saving to ./internal/ml/model_selection/exp_result/benchmark_{dataset}")
draw_structure_data_anytime(
    all_lines=all_lines,
    dataset=datasetfg_name,
    name_img=f"./internal/ml/model_selection/exp_result/benchmark_{dataset}",
    max_value=mx_value,
    figure_size=figure_size,
    annotations=annotations,
    y_ticks=y_lim,
    x_ticks=x_lim,
    unique_labels=["Global Best AUC","BOHB", "RL", "RS", "RE" ],
    x_label_name="Number of explored architectures"
)
