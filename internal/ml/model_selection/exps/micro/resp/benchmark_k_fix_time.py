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

    from src.eva_engine.run_ms import RunModelSelection

    # for this exp, we repeat 100 times and set max to 1000 mins
    total_run = 5

    rms = RunModelSelection(args.search_space, args, is_simulate=True)

    fix_time_budget = 100 * 60
    # approximately on a log scale

    # run two phase with coordinator
    _, best_arch_performance_coord, _, _, _, _, _, _ = \
        rms.select_model_online(
            budget=fix_time_budget,
            data_loader=[None, None, None],
            only_phase1=False,
            run_workers=1)

    # if explore ll model, then no need to explore again in the layer K
    explore_all_cache = []

    # 200: 0.9802417809754849
    for K in [1, 2, 5, 10, 20, 50, 100, 200]:
        time_per_epoch = rms.profile_refinement()
        # calculate time (simulate)
        _, _, B2_actual_epoch_use , _= \
            rms.refinement_phase(
                U=1,
                k_models=K*["512-512-512-512"],
                train_time_per_epoch=time_per_epoch)

        # get time left for filtering -> get N
        refinement_time = B2_actual_epoch_use * rms.sh.time_per_epoch
        time_left_for_filtering = fix_time_budget - refinement_time

        if time_left_for_filtering <= 0:
            print(f"********** fix_time_budget={fix_time_budget}, K={K}, "
                  f"refinement_time needs {refinement_time}, B2_actual_epoch_use needs {B2_actual_epoch_use}, fucked. ")
            continue

        # run two phase w/o coordinator
        N = time_left_for_filtering / rms.profile_filtering()
        print(f"run with fix_time_budget={fix_time_budget}, K={K}, N={N}")

        # already explore ll
        if N >= len(rms.search_space_ins) and len(explore_all_cache) > 0:
            k_models = explore_all_cache[-K:]
        else:
            k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id = rms.filtering_phase(
                N=N, K=K)
            # cache it. low -> high, cache all models.
            explore_all_cache = all_models

        # real refinement phase (simulate)
        p2_best_arch, p2_best_arch_performance, p2_actual_epoch_use, _ = \
            rms.sh.run_phase2(1, k_models)

        print(f"K={K}, AUC={p2_best_arch_performance}, AUC (ATLAS) = {best_arch_performance_coord}")

        # (0.9798742301833366,
        #  K=1, AUC=0.9788888113938895, AUC (ATLAS) = 0.9805218764664727
        # run with fix_time_budget=6000, K=2, N=283125.7647915072
        # K=2, AUC=0.9794521659560285, AUC (ATLAS) = 0.9805218764664727
        # run with fix_time_budget=6000, K=5, N=282399.4107822884
        # K=5, AUC=0.9794521659560285, AUC (ATLAS) = 0.9805218764664727
        # run with fix_time_budget=6000, K=10, N=279009.75873926707
        # K=10, AUC=0.9800915384443517, AUC (ATLAS) = 0.9805218764664727
        # run with fix_time_budget=6000, K=20, N=270051.3926255679
        # K=20, AUC=0.9802417809754849, AUC (ATLAS) = 0.9805218764664727
        # run with fix_time_budget=6000, K=50, N=248987.12635822117
        # K=50, AUC=0.980094755736017, AUC (ATLAS) = 0.9805218764664727
        # run with fix_time_budget=6000, K=100, N=197658.10970675558
        # K=100, AUC=0.9802321101902912, AUC (ATLAS) = 0.9805218764664727
        # run with fix_time_budget=6000, K=200, N=105620.26399204894
        # K=200, AUC=0.9802417809754849, AUC (ATLAS) = 0.9805218764664727

