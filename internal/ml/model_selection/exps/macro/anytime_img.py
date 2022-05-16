import os
import random
from exps.shared_args import parse_arguments
from exps.draw_img_lib import get_plot_compare_with_base_line_cfg

"""
This script is to read the pre-computed result from sqllite, and then run scheduling.
please run other code to get the pre-computed result.
"""

if __name__ == "__main__":
    random.seed(10)
    args = parse_arguments()
    os.environ.setdefault("base_dir", args.base_dir)
    from src.eva_engine.run_ms import RunModelSelection
    from src.query_api.interface import SimulateTrain
    from src.query_api.img_train_baseline import post_processing_train_base_result
    from src.tools.io_tools import write_json

    # this is for acquire the final acc
    fgt = SimulateTrain(space_name=args.search_space)

    saved_dict = {}

    # phase1 + phase2
    run_range_, budget_array, sub_graph_y1, sub_graph_y2, sub_graph_split, draw_graph = \
        get_plot_compare_with_base_line_cfg(args.search_space, args.dataset, False)

    runner = RunModelSelection(args.search_space, args, is_simulate=True)
    y_acc_list_arr = []
    for run_id in run_range_:
        print(f"getting system-based result run={run_range_}")
        y_each_run = []
        for Tmin in budget_array:
            best_arch, _, _, _ = runner.select_model_simulate(Tmin * 60, run_id, only_phase1=False)
            acc_sh_v, _ = fgt.get_ground_truth(arch_id=best_arch, dataset=args.dataset, epoch_num=None)
            y_each_run.append(acc_sh_v)
        y_acc_list_arr.append(y_each_run)

    # phase1
    run_range_, budget_array_p1, sub_graph_y1, sub_graph_y2, sub_graph_split, draw_graph = \
        get_plot_compare_with_base_line_cfg(args.search_space, args.dataset, True)

    runner.is_simulate = True
    y_acc_list_arr_p1 = []
    for run_id in run_range_:
        print(f"getting p1 result run={run_range_}")
        y_each_run_p1 = []
        for Tmin in budget_array_p1:
            # phase1
            best_arch_p1, _, _, _ = runner.select_model_simulate(Tmin * 60, run_id, only_phase1=True)
            # 4. Training it and getting the real accuracy.rain and get the final acc
            acc_sh_v_p1, _ = fgt.get_ground_truth(arch_id=best_arch_p1, dataset=args.dataset, epoch_num=None)
            y_each_run_p1.append(acc_sh_v_p1)
        y_acc_list_arr_p1.append(y_each_run_p1)

    # traing-based ms
    # todo: update the parse result with result["baseline_acc"] and result["baseline_time_budget"] as in train_with_ea
    x_acc_train, y_acc_train_l, y_acc_train_m, y_acc_train_h = post_processing_train_base_result(
        search_space=args.search_space, dataset=args.dataset)

    # 2 phase ms
    saved_dict["y_acc_list_arr"] = y_acc_list_arr
    saved_dict["x_T_list"] = budget_array

    # training-based with ea
    saved_dict["x_acc_train"] = x_acc_train
    saved_dict["y_acc_train_l"] = y_acc_train_l
    saved_dict["y_acc_train_m"] = y_acc_train_m
    saved_dict["y_acc_train_h"] = y_acc_train_h

    # training-free ms
    saved_dict["y_acc_list_arr_only_phase1"] = y_acc_list_arr_p1
    saved_dict["x_T_list_only_phase1"] = budget_array_p1

    write_json(f"{args.result_dir}/0_macro_res_{args.search_space}_{args.dataset}", saved_dict)
