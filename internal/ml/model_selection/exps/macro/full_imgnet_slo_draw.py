from typing import List

from src.tools.compute import sample_in_log_scale_new
from src.tools.io_tools import read_json
from exps.draw_tab_lib import draw_structure_data_anytime, draw_structure_data_anytime_system_version_imageNetFULL


def get_dataset_parameters(dataset):
    parameters = {
        "ImageNet16-120": {
            "epoch": 25,
            "train_based_re": "./internal/ml/model_selection/exp_result/train_base_line_re_ImageNet16-120_epoch_200.json",
            # "sys_end2end_res": "./internal/ml/model_selection/exp_result/imagenetfull/res_end_2_end_nasbench201_ImageNetfull_jacflow.json",
            "sys_end2end_res": "./internal/ml/model_selection/exp_result/imagenetfull/res_end_2_end_nasbench201_ImageNet16-120_-1_12_jacflow.json",
            "mx_value": 47,
            "x_lim": [None, 3800*24*60],
            "y_lim": [33, 48.5],
            "figure_size": (6.8, 4.8),
            "datasetfg_name": dataset,
            "annotations": [
                ["AmoebaNet-A", 74.5, 4536000],
                ["NASNet-A", 74, 2880000],
                ["TE-NAS", 74.1, 288],
            ],
            "remove_n_points": 0,
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
    trian_time_per_model = params['epoch'] * 96080 / 60

    result_dir = "./internal/ml/model_selection/exp_result/"

    system_result = read_json(params['sys_end2end_res'])
    train_based_res = read_json(params["train_based_re"])

    # here we record the time usage, in minuts
    sampled_train_x, sampled_train_y = sample_some_points(
        x_array=[
            [trian_time_per_model * i for i in range(1, 1 + len(train_based_res["sys_acc"][0]))]  # number of samples
            for _ in range(len(train_based_res)) # number of run
        ],
        y_2d_array=train_based_res["sys_acc"],
        save_points=7,
        remove_n_points=0)

    # here we record the time usage, in minuts
    sampled_sys_x, sampled_sys_y = sample_some_points(
        x_array=[system_result["sys_time_budget"] for _ in system_result["sys_acc"]],
        y_2d_array=system_result["sys_acc"],
        save_points=12,
        remove_n_points=1)

    all_lines = [
        [sampled_train_x, sampled_train_y],  # train-based
        [sampled_sys_x, sampled_sys_y],  # two-phase
    ]

    draw_structure_data_anytime_system_version_imageNetFULL(
        all_lines=all_lines,
        dataset=params['datasetfg_name'],
        name_img=f"{result_dir}/anytime_{dataset}_full",
        max_value=params['mx_value'],
        figure_size=params['figure_size'],
        annotations=params['annotations'],
        y_ticks=params['y_lim'],
        x_ticks=params['x_lim']
    )


from src.common.constant import Config

# Choose dataset to process
dataset = Config.imgNet

generate_and_draw_data(dataset)
