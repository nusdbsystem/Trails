from exps.shared_args import parse_arguments
import random
import calendar
import os
import time
import matplotlib

args = parse_arguments()

gmt = time.gmtime()
ts = calendar.timegm(gmt)
os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
os.environ.setdefault("base_dir", args.base_dir)
from src.query_api.interface import SimulateScore
from src.eva_engine.phase2.evaluator import P2Evaluator
from src.search_space.init_search_space import init_search_space
from src.eva_engine.run_ms import RunModelSelection
from src.query_api.interface import profile_NK_trade_off

matplotlib.use('TkAgg')  # Or any other supported backend
from matplotlib import pyplot as plt
import numpy as np


def random_combination(iterable, sample_size):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)


def run_evolution_search(max_trained_models=1000,
                         pool_size=64,
                         tournament_size=10,
                         zero_cost_warmup=0,
                         zero_cost_move=False,
                         tfmem="express_flow",
                         dataset_name="frappe",
                         query_epoch=19,
                         args=None):
    best_tests = [0.0]
    pool = []  # (validation, spec) tuples
    time_usage_lst = [0]
    num_trained_models = 0

    search_space_ins = init_search_space(args)

    _evaluator = P2Evaluator(search_space_ins, dataset_name,
                             is_simulate=True,
                             train_loader=None,
                             val_loader=None)

    score_getter = SimulateScore(
        space_name=search_space_ins.name,
        dataset_name=dataset_name)

    # fill the initial pool
    cur_time_used = 0
    if zero_cost_warmup > 0:
        zero_cost_pool = []
        for _ in range(zero_cost_warmup):
            arch_id, arch_micro = search_space_ins.random_architecture_id()

            score = score_getter.query_all_tfmem_score(arch_id=arch_id)
            zero_cost_pool.append((score[tfmem], arch_id, arch_micro))
            zero_cost_pool = sorted(zero_cost_pool, key=lambda i: i[0], reverse=True)

        # already score 3k models, here
        cur_time_used += zero_cost_warmup * score_getter.api.get_score_one_model_time("gpu")

    for i in range(pool_size):
        if zero_cost_warmup > 0:
            arch_id = zero_cost_pool[i][1]
            arch_micro = zero_cost_pool[i][2]
        else:
            arch_id, arch_micro = search_space_ins.random_architecture_id()

        full_train_auc, time_usage = _evaluator.p2_evaluate(str(arch_id), query_epoch)
        # get accuracy by training.
        num_trained_models += 1
        pool.append((full_train_auc, arch_micro))

        cur_time_used += time_usage
        time_usage_lst.append(cur_time_used / 60)

        if full_train_auc > best_tests[-1]:
            best_tests.append(full_train_auc)
        else:
            best_tests.append(best_tests[-1])

    # After the pool is seeded, proceed with evolving the population.
    while True:

        # sample some value from the pool
        sample = random_combination(pool, tournament_size)
        best_arch_spec = sorted(sample, key=lambda i: i[0])[-1][1]
        if zero_cost_move:

            _all_combs = search_space_ins.mutate_architecture_move_proposal(best_arch_spec)
            all_combs = []
            for ele in _all_combs:
                arch_id = ele[0]
                arch_micro = ele[1]
                score = score_getter.query_all_tfmem_score(arch_id=arch_id)
                all_combs.append((score[tfmem], arch_id, arch_micro))

            if random.random() > 0.75:
                best_arch_spec = random.choice(all_combs)
            else:
                best_arch_spec = sorted(list(all_combs), key=lambda i: i[0])[-1]

            _, arch_id, arch_micro = best_arch_spec

            cur_time_used += len(_all_combs) * score_getter.api.get_score_one_model_time("gpu")

        else:
            arch_id, arch_micro = search_space_ins.mutate_architecture(best_arch_spec)

        # train model to get auc
        full_train_auc, time_usage = _evaluator.p2_evaluate(str(arch_id), query_epoch)
        num_trained_models += 1

        cur_time_used += time_usage
        time_usage_lst.append(cur_time_used / 60)

        # kill the oldest individual in the population.
        pool.append((full_train_auc, arch_micro))
        pool.pop(0)

        if full_train_auc > best_tests[-1]:
            best_tests.append(full_train_auc)
        else:
            best_tests.append(best_tests[-1])

        if num_trained_models >= max_trained_models:
            break
    # return time in mins
    return time_usage_lst, best_tests


def filter_refinment_fully_training(dataset_name="frappe",
                                    query_epoch=19):
    print("Running filter_refinment_fully_training...")

    rms = RunModelSelection(args.search_space, args, is_simulate=True)
    search_space_ins = init_search_space(args)
    _evaluator = P2Evaluator(search_space_ins, dataset_name,
                             is_simulate=True,
                             train_loader=None,
                             val_loader=None)

    score_time_per_model = rms.profile_filtering()
    n_k_ratio = profile_NK_trade_off(dataset_name)

    total_run = 50
    auc_lst = []
    time_usage_lst = []
    for run_id in range(total_run):
        _time_usage_lst = []
        _auc_lst = []
        for N in range(30, 3030, 50):
            cur_time_used = 0
            K = int(N / n_k_ratio)
            k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id = rms.filtering_phase(
                N=N, K=K)
            cur_time_used += N * score_time_per_model
            if K > 1:
                train_auc = []
                for arch_id in k_models:
                    full_train_auc, time_usage = _evaluator.p2_evaluate(str(arch_id), query_epoch)
                    cur_time_used += time_usage
                    train_auc.append(full_train_auc)
            else:
                train_auc = []
                full_train_auc, time_usage = _evaluator.p2_evaluate(str(k_models[0]), query_epoch)
                train_auc.append(full_train_auc)

            _auc_lst.append(max(train_auc))
            _time_usage_lst.append(cur_time_used / 60)
        auc_lst.append(_auc_lst)
        time_usage_lst.append(_time_usage_lst)

    return [time_usage_lst, auc_lst, "Filtering + Refinement (Fully Train)"]


def export_warm_up_move_proposal(tfmem: str):
    print(f"Running warm_up_move_proposal using {tfmem}...")
    total_run = 50
    ae_warmup_time, ae_warmup = [], []
    for run_id in range(total_run):
        _ae_warmup_time, _ae_warmup = run_evolution_search(
            max_trained_models=100,
            pool_size=64,
            tournament_size=10,
            zero_cost_warmup=3000,
            zero_cost_move=False,
            tfmem=tfmem,
            dataset_name="frappe",
            query_epoch=19,
            args=args)

        ae_warmup_time.append(_ae_warmup_time)
        ae_warmup.append(_ae_warmup)

    ae_move_time, ae_move = [], []
    for run_id in range(total_run):
        _ae_move_time, _ae_move = run_evolution_search(
            max_trained_models=100,
            pool_size=64,
            tournament_size=10,
            zero_cost_warmup=0,
            zero_cost_move=True,
            tfmem=tfmem,
            dataset_name="frappe",
            query_epoch=19,
            args=args)

        ae_move_time.append(_ae_move_time)
        ae_move.append(_ae_move)

    return [ae_warmup_time, ae_warmup, f"EA + warmup-{tfmem} (3K)"], \
           [ae_move_time, ae_move, f"EA + move-{tfmem}"]


def plot_experiment(exp_list, title):
    def plot_exp(time_usg, exp, label):
        exp = np.array(exp)
        q_75_y = np.quantile(exp, .75, axis=0)
        q_25_y = np.quantile(exp, .25, axis=0)
        mean_y = np.mean(exp, axis=0)
        time_usg = np.array(time_usg)
        q_75_time = np.quantile(time_usg, .75, axis=0)
        q_25_time = np.quantile(time_usg, .25, axis=0)
        mean_time = np.mean(time_usg, axis=0)
        plt.plot(mean_time, mean_y, "-*", label=label, )
        plt.fill_between(mean_time, q_25_y, q_75_y, alpha=0.1)

        # exp = np.array(exp)
        # q_75 = np.quantile(exp, .75, axis=0)
        # q_25 = np.quantile(exp, .25, axis=0)
        # mean = np.mean(exp, axis=0)
        # plt.plot(mean, label=label)
        # plt.fill_between(range(len(q_25)), q_25, q_75, alpha=0.1)

    fig, ax = plt.subplots()

    for time_usg, exp, ename in exp_list:
        plot_exp(time_usg, exp, ename)
    plt.grid()
    plt.xlabel('Time in mins')
    plt.ylabel('Test Accuracy')
    plt.ylim(0.9700, 0.9815)

    plt.xscale("log")
    plt.xlim([0.01, 150])

    plt.legend()
    plt.title(title)
    fig.savefig(f"./internal/ml/model_selection/exp_result/warm_up_move_proposal.pdf",
                bbox_inches='tight')


if __name__ == "__main__":
    # filter_full_train = filter_refinment_fully_training()
    warmups, move = export_warm_up_move_proposal("express_flow")
    warmups2, move2 = export_warm_up_move_proposal("synflow")
    warmups3, move3 = export_warm_up_move_proposal("nas_wot")

    all_lines = [
        # warmups,
        move, move2, move3,
        # filter_full_train
    ]

    plot_experiment(all_lines, 'EA Search')
    print("Done")
