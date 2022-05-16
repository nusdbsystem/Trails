import numpy as np
from src.common.constant import Config
from exps.baseline.nas_alg import get_base_annotations
from src.tools.compute import sample_in_log_scale_new
from src.tools.io_tools import read_json
from exps.draw_img_lib import get_plot_compare_with_base_line_cfg

# dataset = Config.c10
dataset = Config.c100
# dataset = Config.imgNet
search_space = Config.NB201

img_in_graph = "ImageNet" if dataset == Config.imgNet else dataset
result_dir = "./internal/ml/model_selection/exp_result"
saved_dict = read_json(f"{result_dir}/jacflow/0_macro_res_{search_space}_{dataset}")

run_range_, budget_array, sub_graph_y1, sub_graph_y2, sub_graph_split, draw_graph = get_plot_compare_with_base_line_cfg(
    search_space, dataset, True)


def process_saved_dict(dictionary, keys):
    return (dictionary[key] for key in keys)


y_acc_list_arr, x_T_list, x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h, y_acc_list_arr_only_phase1, x_T_list_only_phase1 = \
    process_saved_dict(saved_dict,
                       ["y_acc_list_arr", "x_T_list", "x_acc_train", "y_acc_train_l", "y_acc_train_m", "y_acc_train_h",
                        "y_acc_list_arr_only_phase1", "x_T_list_only_phase1"])

baseline = read_json(f"{result_dir}/train_base_line_re_{dataset}_epoch_200.json")

time_used = np.array(baseline['sys_time_budget'])
x_acc_train = np.quantile(time_used, 0.5, axis=0).tolist()

acc_reached = np.array(baseline['sys_acc'])
y_acc_train_l = np.quantile(acc_reached, 0.25, axis=0).tolist()
y_acc_train_m = np.quantile(acc_reached, 0.5, axis=0).tolist()
y_acc_train_h = np.quantile(acc_reached, 0.75, axis=0).tolist()


def sample_array_elements(array, indexes):
    return [array[i] for i in indexes]


def process_arrays(time_array, value_array, value_p1_array, train_x_time_array, train_x_acc_array_low,
                   train_x_acc_array_mean, train_x_acc_array_high):
    new_index_array = sample_in_log_scale_new(time_array, 6)
    new_time_array = sample_array_elements(time_array, new_index_array)
    new_value_array = [sample_array_elements(ori_value, new_index_array) for ori_value in value_array]
    new_value_p1_array = [sample_array_elements(ori_value, new_index_array) for ori_value in value_p1_array]

    new_index_array_train_time = sample_in_log_scale_new(train_x_time_array, 6)
    new_train_x_time_array = sample_array_elements(train_x_time_array, new_index_array_train_time)
    new_train_x_acc_array_low = sample_array_elements(train_x_acc_array_low, new_index_array_train_time)
    new_train_x_acc_array_mean = sample_array_elements(train_x_acc_array_mean, new_index_array_train_time)
    new_train_x_acc_array_high = sample_array_elements(train_x_acc_array_high, new_index_array_train_time)

    return new_time_array, new_value_array, new_value_p1_array, new_train_x_time_array, new_train_x_acc_array_low, new_train_x_acc_array_mean, new_train_x_acc_array_high


def sample_some_points(dataset, *args):
    max_performance_dict = {Config.c10: 94.3, Config.c100: 72.8, "other": 47}
    max_performance = max_performance_dict.get(dataset, max_performance_dict["other"])
    return (*process_arrays(*args), max_performance)


x_T_list, y_acc_list_arr, y_acc_list_arr_only_phase1, x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h, max_performance = sample_some_points(
    dataset, x_T_list, y_acc_list_arr, y_acc_list_arr_only_phase1, x_acc_train, y_acc_train_l, y_acc_train_m,
    y_acc_train_h)


def get_dataset_y_label(inout_data):
    if inout_data == Config.c10:
        dataset = "C10"
    elif inout_data == Config.c100:
        dataset = "C100"
    else:
        dataset = "IN16"
    return dataset


draw_graph(result_dir, y_acc_list_arr, x_T_list, y_acc_list_arr_only_phase1, x_T_list,
           x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h,
           get_base_annotations(dataset), sub_graph_split,
           f"{search_space}_{dataset}", img_in_graph, max_performance,
           sub_graph_y1, sub_graph_y2, get_dataset_y_label(dataset))
