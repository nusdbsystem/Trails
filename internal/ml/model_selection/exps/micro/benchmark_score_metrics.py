# this is the main function of model selection.

import calendar
import os
import time

from exps.shared_args import parse_arguments

if __name__ == "__main__":
    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.query_api.interface import SimulateTrain, SimulateScore
    from src.search_space.init_search_space import init_search_space
    from src.tools.io_tools import write_json, read_json
    from src.eva_engine.phase1.run_phase1 import RunPhase1, p1_evaluate_query

    # configurable settings for benchmarking
    only_phase1 = True
    total_run = 50

    checkpoint_name = f"{args.result_dir}/re_{args.dataset}_{args.kn_rate}_{args.num_points}_auc.json"
    checkpoint_score_name = f"{args.result_dir}/re_{args.dataset}_{args.kn_rate}_{args.num_points}_score.json"

    result = {}
    result["explored_arch"] = []
    result["achieved_value"] = []

    resultscore = {}
    resultscore["explored_arch"] = []
    resultscore["achieved_value"] = []

    search_space_ins = init_search_space(args)
    acc_getter = SimulateTrain(space_name=search_space_ins.name)

    for run_id in range(total_run):
        run_begin_time = time.time()

        p1_runner = RunPhase1(
            args=args,
            K=1, N=5000,
            search_space_ins=search_space_ins,
            is_simulate=True)

        K_models, _, current_highest_score, current_models_perforamnces = p1_runner.run_phase1()

        arch_perform_lst = []
        for ele in current_models_perforamnces:
            acc, time_usage = acc_getter.get_ground_truth(
                arch_id=ele, epoch_num=args.epoch, dataset=args.dataset)
            arch_perform_lst.append(acc)

        result["explored_arch"].append(list(range(1, len(arch_perform_lst) + 1)))
        result["achieved_value"].append(arch_perform_lst)

        resultscore["explored_arch"].append(list(range(1, len(current_highest_score) + 1)))
        resultscore["achieved_value"].append(current_highest_score)

        print(f"finish run_id = {run_id}, using {time.time() - run_begin_time}")

        # checkpointing each run
    write_json(checkpoint_name, result)
    write_json(checkpoint_score_name, resultscore)
