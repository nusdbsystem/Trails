from typing import List

from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from src.common.constant import Config

from src.tools.compute import sample_in_log_scale_new
from exps.draw_tab_lib import draw_structure_data_anytime
from src.tools.io_tools import read_json

# dataset = "frappe"
# dataset = "ImageNet16-120"
# dataset = "cifar10"
dataset = "cifar100"


if dataset == Config.imgNet:
    img_in_graph = "ImageNet"
elif dataset == Config.c10:
    img_in_graph = "CIFAR10"
elif dataset == Config.c100:
    img_in_graph = "CIFAR100"
else:
    pass

result_dir = "./internal/ml/model_selection/exp_result/"
saved_file = f"{result_dir}/micro_phase2_{dataset}"

if dataset == Config.Frappe:
    sub_graph_y1 = [98, 98.05]

if dataset == Config.Criteo:
    sub_graph_y1 = None

if dataset == Config.c10:
    # C10 array
    sub_graph_y1 = [93.60, 94.1]
elif dataset == Config.c100:
    # C100 array
    sub_graph_y1 = [70.5, 72.5]
elif dataset == Config.imgNet:
    # ImgNet X array
    sub_graph_y1 = [44, 46.4]

result_save_dic = read_json(saved_file)
fig2 = plt.figure(figsize=(6.4, 3.8))


def sample_some_points(x_array, y_2d_array, save_points, remove_n_points=1) -> (List, List):
    # result_x_array = []
    # result_y_array = []
    # for run_id, time_list in enumerate(x_array):
    #     save_points = min(len(time_list), save_points)
    #     indices = sample_in_log_scale_new(time_list, save_points)
    #     # Sample the list using the calculated indices
    #     each_run_x_array = [time_list[i] for i in indices]
    #     each_run_y_array = [y_2d_array[run_id][int(i)] for i in indices]
    #
    #     if remove_n_points != 0:
    #         result_x_array.append(each_run_x_array[:-remove_n_points])
    #         result_y_array.append(each_run_y_array[:-remove_n_points])
    #     else:
    #         result_x_array.append(each_run_x_array)
    #         result_y_array.append(each_run_y_array)

    result_x_array = []
    result_y_array = []
    for run_id in range(len(x_array)):
        x_time = x_array[run_id]
        y_acc = y_2d_array[run_id]
        index = 0
        for i in range(len(y_acc)):
            if y_acc[i] == 0:
                index += 1
            else:
                break
        x_time = x_time[index:]
        y_acc = y_acc[index:]

        result_x_array.append(x_time)
        result_y_array.append(y_acc)

    return result_x_array, result_y_array


mark_list = ["o-", "*-", "<-"]
key_list = ["uniform", "sr", "sh"]

all_lines = []
save_points = 11
for key in key_list:
    if key not in result_save_dic:
        continue
    value = result_save_dic[key]
    time_used = value["time_used"]
    acc_reached = value["acc_reached"]
    if key == "sh":
        The_name = "SUCCHALF"
        sampled_train_x, sampled_train_y = sample_some_points(
            x_array=time_used,
            y_2d_array=acc_reached,
            save_points=save_points,
            remove_n_points=0)
    elif key == "uniform":
        The_name = "UNIFORM"
        sampled_train_x, sampled_train_y = sample_some_points(
            x_array=time_used,
            y_2d_array=acc_reached,
            save_points=save_points,
            remove_n_points=0)
    else:
        The_name = "SUCCREJCT"
        sampled_train_x, sampled_train_y = sample_some_points(
            x_array=time_used,
            y_2d_array=acc_reached,
            save_points=save_points,
            remove_n_points=0)
    inner_res = [sampled_train_x, sampled_train_y, The_name]
    all_lines.append(inner_res)

print(f"saving to {result_dir}/micro_phase2_{dataset}")

draw_structure_data_anytime(
    all_lines=all_lines,
    dataset="C100",
    name_img=f"{result_dir}/mmicro_phase2_{dataset}",
    max_value=-1,
    y_ticks=sub_graph_y1,
    x_ticks=None
)

# export_legend(fig2, filename="phase2_micro_legend", unique_labels=["UNIFORM", "SUCCREJCT", "SUCCHALF"])
# fig2.savefig(f"phase2_micro_{dataset}.pdf", bbox_inches='tight')
