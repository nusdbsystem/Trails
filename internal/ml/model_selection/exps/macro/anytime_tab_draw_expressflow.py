from typing import List

from src.tools.compute import sample_in_log_scale_new
from src.tools.io_tools import read_json
from exps.draw_tab_lib import draw_structure_data_anytime, draw_structure_data_anytime_system_version


def get_dataset_parameters(dataset):
    parameters = {
        "frappe": {
            "epoch": 13,
            "sys_end2end_res": "./internal/ml/model_selection/exp_result/expressflow/res_end_2_end_mlp_sp_frappe_-1_7_express_flow.json",
            "sys_end2end_p1": "./internal/ml/model_selection/exp_result/jacflow/res_end_2_end_frappe_100_12_p1.json",
            "tab_nas_res": "./internal/ml/model_selection/exp_result/tabNAS_benchmark_frappe_epoch_13.json",
            "train_based_re": "./internal/ml/model_selection/exp_result/train_base_line_re_frappe_epoch_13.json",
            "mx_value": 98.14,
            "x_lim": [0.01, None],
            "y_lim": [97.77, 98.18],
            "figure_size": (6.2, 4),
            "datasetfg_name": dataset,
            "annotations": [],
            "remove_n_points": 2,
        },
        "uci_diabetes": {
            "epoch": 0,
            "sys_end2end_res": "./internal/ml/model_selection/exp_result/expressflow/res_end_2_end_mlp_sp_uci_diabetes_-1_10_express_flow.json",
            "sys_end2end_p1": "./internal/ml/model_selection/exp_result/jacflow/res_end_2_end_uci_diabetes_100_12_p1.json",
            "tab_nas_res": "./internal/ml/model_selection/exp_result/tabNAS_benchmark_uci_diabetes_epoch_0.json",
            "train_based_re": "./internal/ml/model_selection/exp_result/train_base_line_re_uci_diabetes_epoch_0.json",
            "mx_value": 67.47755324313862,
            "x_lim": [0.01, 300],
            "y_lim": [61.811, 68],
            "figure_size": (6.2, 4),
            "datasetfg_name": "Diabetes",
            "annotations": [],
            "remove_n_points": 2,
        },

        "criteo": {
            "epoch": 9,
            "sys_end2end_res": "./internal/ml/model_selection/exp_result/expressflow/res_end_2_end_mlp_sp_criteo_-1_6_express_flow.json",
            "sys_end2end_p1": "./internal/ml/model_selection/exp_result/jacflow/res_end_2_end_criteo_100_12_p1.json",
            "tab_nas_res": "./internal/ml/model_selection/exp_result/tabNAS_benchmark_criteo_epoch_9.json",
            "train_based_re": "./internal/ml/model_selection/exp_result/train_base_line_re_criteo_epoch_9.json",
            "mx_value": 80.32615745641593,
            "x_lim": [0.01, 5000],
            "y_lim": [80.121, 80.349],
            "figure_size": (6.2, 4),
            "datasetfg_name": dataset,
            "annotations": [],
            "remove_n_points": 1,
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


def generate_and_draw_data(dataset):
    params = get_dataset_parameters(dataset)
    if not params:
        print(f"No parameters for the dataset: {dataset}")
        return

    # to min
    trian_time = ((params['epoch'] + 1) * api.get_train_one_epoch_time("gpu")) / 60

    result_dir = "./internal/ml/model_selection/exp_result/"

    system_result = read_json(params['sys_end2end_res'])
    system_p1_result = read_json(params['sys_end2end_p1'])
    tab_nas_res = read_json(params["tab_nas_res"])
    train_based_res = read_json(params["train_based_re"])

    # here we record the time usage, in minuts
    sampled_train_x, sampled_train_y = sample_some_points(
        x_array=train_based_res["sys_time_budget"],
        y_2d_array=train_based_res["sys_acc"],
        save_points=7,
        remove_n_points=0)

    # here we record the time usage, in minuts
    sampled_sys_x, sampled_sys_y = sample_some_points(
        x_array=[system_result["sys_time_budget"] for _ in system_result["sys_acc"]],
        y_2d_array=system_result["sys_acc"],
        save_points=7,
        remove_n_points=0)

    # here we record number of arch explored
    try:
        tabnas_x, tabnas_y = sample_some_points(
            x_array=[[earch * trian_time for earch in ele] for ele in tab_nas_res['baseline_time_budget']],
            y_2d_array=tab_nas_res['baseline_acc'],
            save_points=100,
            remove_n_points=0)
    except:
        tabnas_x, tabnas_y = sample_some_points(
            x_array=[[earch * trian_time for earch in ele] for ele in tab_nas_res["sys_time_budget"]],
            y_2d_array=tab_nas_res["sys_acc"],
            save_points=100,
            remove_n_points=0)

    all_lines = [
        [sampled_train_x, sampled_train_y, "RE-NAS"],
        # [system_p1_result["sys_time_budget"], system_p1_result["sys_acc"], "Training-Free MS"],
        [sampled_sys_x, sampled_sys_y, "ATLAS"],
        [tabnas_x, tabnas_y, "TabNAS"],
    ]

    draw_structure_data_anytime(
        all_lines=all_lines,
        dataset=params['datasetfg_name'],
        name_img=f"{result_dir}/anytime_{dataset}_v1",
        max_value=params['mx_value'],
        figure_size=params['figure_size'],
        annotations=params['annotations'],
        y_ticks=params['y_lim'],
        x_ticks=params['x_lim']
    )


# Choose dataset to process
# dataset = "frappe"
dataset = "uci_diabetes"
# dataset = "criteo"

from src.query_api.query_api_mlp import GTMLP

api = GTMLP(dataset)
generate_and_draw_data(dataset)
