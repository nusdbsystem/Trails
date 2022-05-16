# this is the main function of model selection.
import numpy as np
import calendar
import os
import time
from exps.shared_args import parse_arguments
from multiprocessing import Process


def run_with_time_budget(time_budget: float, only_phase1: bool):
    """
    :param time_budget: the given time budget, in second
    :return:
    """

    best_arch, best_arch_performance, time_usage, _, _, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id = \
        rms.select_model_online(
            budget=time_budget,
            data_loader=[None, None, None],
            only_phase1=only_phase1,
            run_workers=1)

    return best_arch, best_arch_performance, time_usage, \
           all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id


def select_top_k_items(list_m, k):
    unique_list = []
    for item in list_m:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list[-k:]


def draw_graph(scaled_data, two_d_epoch, u_ticks, k_ticks, dataset):
    from exps.draw_tab_lib import plot_heatmap

    plot_heatmap(data=scaled_data,
                 fontsize=18,
                 x_array_name="U(# Training Epoch)",
                 y_array_name="K(# Explored Architecture)",
                 title="Accuracy Achieved",
                 output_file=f"./internal/ml/model_selection/exp_result/micro_ku_accuracy_{dataset}.pdf",
                 decimal_places=2,
                 u_ticks=u_ticks,
                 k_ticks=k_ticks)

    plot_heatmap(data=two_d_epoch,
                 fontsize=18,
                 x_array_name="U(# Training Epoch)",
                 y_array_name="K(# Explored Architecture)",
                 title="Time Usage",
                 output_file=f"./internal/ml/model_selection/exp_result/micro_ku_epochs_{dataset}.pdf",
                 decimal_places=1,
                 u_ticks=u_ticks,
                 k_ticks=k_ticks)


def convert_to_two_dim_list(original_list, len_k, len_u):
    """
    Device the original_list into len_k sub-list, each with len_u elements
    Return a two dimension list
    """
    if len_k * len_u > len(original_list):
        print("Error: len_k * len_u > len(original_list). Cannot proceed.")
        return
    two_dim_list = [original_list[i * len_u: (i + 1) * len_u] for i in range(len_k)]
    return two_dim_list


if __name__ == "__main__":
    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.eva_engine.run_ms import RunModelSelection
    from src.common.constant import Config
    from src.logger import logger

    # for this exp, we repeat 100 times and set max to 1000 mins
    total_run = 20

    rms = RunModelSelection(args.search_space, args, is_simulate=True)

    # Fix budget to 100 mins and only use phase1, try differetn K and U
    if args.dataset in [Config.c10, Config.c100, Config.imgNet]:
        budget_array = [300]
        k_options = [2, 4, 8, 16]
        u_options = [1, 4, 16, 64, 200]
    else:
        budget_array = [10]
        k_options = [2, 4, 8, 16]
        u_options = [1, 2, 4, 8, 16]

    exp_run = {
        "total_epoch": [],
        "models": [],
    }
    # todo: to run the uci dataset, update t_acc = self.mlp_train[self.dataset][arch_id][str(epoch_num-1)]["valid_auc"] in query_api_mlp.py
    for run_id in range(total_run):
        phase2_total_training_epoches = []
        phase2_model_performance_find = []
        run_begin_time = time.time()
        run_acc_list = []
        for time_budget in budget_array:
            time_budget_sec = time_budget * 60
            logger.info(f"\n Running job with budget={time_budget} min \n")
            best_arch, best_arch_performance, time_usage, \
            all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id = \
                run_with_time_budget(time_budget_sec, only_phase1=True)

            # try various K, U to decide the second phase.
            for k in k_options:
                # select top K
                top_k_models = select_top_k_items(all_models, k)
                print(
                    f"After exploring {len(p1_trace_highest_scored_models_id)}, "
                    f"models, total unique models are {len(set(p1_trace_highest_scored_models_id))}, "
                    f"picking {k} models of size {len(set(top_k_models))} from {len(set(all_models))} models, "
                )
                for u in u_options:
                    p2_best_arch, p2_best_arch_performance, p2_actual_epoch_use, _ = \
                        rms.sh.run_phase2(u, top_k_models)
                    phase2_total_training_epoches.append(p2_actual_epoch_use)
                    phase2_model_performance_find.append(p2_best_arch_performance)
            exp_run["total_epoch"].append(phase2_total_training_epoches)
            exp_run["models"].append(phase2_model_performance_find)

            print(f"finish run_id = {run_id}, using {time.time() - run_begin_time}")

    # Record the medium value
    exp_run["total_epoch"] = np.quantile(np.array(exp_run["total_epoch"]), .5, axis=0).tolist()
    exp_run["models"] = np.quantile(np.array(exp_run["models"]), .5, axis=0).tolist()

    # Draw the two-dimension graph
    two_d_epoch = convert_to_two_dim_list(exp_run["total_epoch"], len(k_options), len(u_options))
    two_d_model_acc = convert_to_two_dim_list(exp_run["models"], len(k_options), len(u_options))
    scaled_data = [[value * 100 if value < 10 else value for value in row] for row in two_d_model_acc]
    print("done")

    print(two_d_epoch)
    print(scaled_data)

    # put your scaled_data and two_d_epoch here
    p = Process(target=draw_graph, args=(scaled_data, two_d_epoch, u_options, k_options, args.dataset))
    p.start()
    p.join()
