import calendar
import os
import time
from exps.shared_args import parse_arguments
from src.tools.compute import log_scale_x_array
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or any other supported backend
from matplotlib import pyplot as plt
import numpy as np


def generate_data_loader():
    if args.dataset in [Config.c10, Config.c100, Config.imgNet]:
        train_loader, val_loader, class_num = dataset.get_dataloader(
            train_batch_size=args.batch_size,
            test_batch_size=args.batch_size,
            dataset=args.dataset,
            num_workers=1,
            datadir=os.path.join(args.base_dir, "data"))
        test_loader = val_loader
    else:
        train_loader, val_loader, test_loader = libsvm_dataloader(
            args=args,
            data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
            nfield=args.nfield,
            batch_size=args.batch_size)
        class_num = args.num_labels

    return train_loader, val_loader, test_loader, class_num


def select_with_noise(all_explored_models, search_space_ins, K, noise_degree=0.1):
    # Number of models to be selected from top K and from the rest
    top_K_count = int(K * (1 - noise_degree))
    rest_count = K - top_K_count

    # Selecting 90% of the top K models
    top_K_selected = random.sample(all_explored_models[-K:], top_K_count)

    # Selecting 10% from the rest of the models
    rest_selected = []
    for _ in range(rest_count):
        while True:
            arch_id, arch_micro = search_space_ins.random_architecture_id()
            if arch_id not in rest_selected:
                rest_selected.append(arch_id)
                break

    # Combining the selected models
    selected_models = top_K_selected + rest_selected
    # print(f" --- sample {len(top_K_selected)} from top K, {len(rest_selected)} from rest")
    return selected_models


def run_with_time_budget(time_budget: float, is_simulate: bool, only_phase1: bool):
    """
    :param time_budget: the given time budget, in second
    :return:
    """

    # define dataLoader, and sample a mini-batch

    rms = RunModelSelection(args.search_space, args, is_simulate=is_simulate)

    score_time_per_model = rms.profile_filtering()
    train_time_per_epoch = rms.profile_refinement()
    K, U, N = rms.coordination(time_budget, score_time_per_model, train_time_per_epoch, only_phase1)
    k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id = rms.filtering_phase(N, K)
    print(f"When time_budget = {time_budget}, Total explored {len(all_models)} models, select top {K} models.")

    degree_auc = []
    for noise_degree in noisy_degree:
        k_models = select_with_noise(all_models, rms.search_space_ins, K, noise_degree)
        best_arch, best_arch_performance, _, _ = rms.refinement_phase(U, k_models)
        degree_auc.append(best_arch_performance)
    return degree_auc


def plot_experiment(exp_list, title):
    def plot_exp(time_usg, exp, label):
        exp = np.array(exp)
        q_75_y = np.quantile(exp, .75, axis=0)
        q_25_y = np.quantile(exp, .25, axis=0)
        mean_y = np.mean(exp, axis=0)

        print(
            f"noisy_degree ={noisy_degree}, "
            f"sys_acc_m={mean_y}, sys_acc_m_25={q_25_y}")

        plt.plot(time_usg, mean_y, "-*", label=label, )
        plt.fill_between(time_usg, q_25_y, q_75_y, alpha=0.1)

    fig, ax = plt.subplots()

    for time_usg, exp, ename in exp_list:
        plot_exp(time_usg, exp, ename)
    plt.grid()

    plt.xscale("log")
    # plt.xlim([0.01, 150])

    plt.xlabel('Time in mins')
    plt.ylabel('Test Accuracy')

    plt.legend()
    plt.title(title)
    fig.savefig(f"./internal/ml/model_selection/exp_result/noisy_inflence.pdf",
                bbox_inches='tight')


if __name__ == "__main__":
    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.eva_engine.run_ms import RunModelSelection
    from src.dataset_utils import dataset
    from src.common.constant import Config
    from src.logger import logger
    from src.dataset_utils.structure_data_loader import libsvm_dataloader
    from src.tools.io_tools import write_json

    train_loader, val_loader, test_loader, class_num = generate_data_loader()
    args.num_labels = class_num

    # configurable settings for benchmarking
    is_simulate = True
    only_phase1 = False

    total_run = 50

    budget_array = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    noisy_degree = [0, 0.3, 0.5, 0.7, 1]
    all_lines_auc = {}
    for ele in noisy_degree:
        all_lines_auc[f"noisy degree - {ele}"] = []

    for run_id in range(total_run):
        # here each run have one list
        for ele in noisy_degree:
            all_lines_auc[f"noisy degree - {ele}"].append([])

        for time_budget in budget_array:
            run_begin_time = time.time()
            time_budget_sec = time_budget * 60
            logger.info(f"\n Running job with budget={time_budget} min \n")
            # _degree_auc: [AUC_noisy1, AUC_noisy2...]
            _degree_auc = run_with_time_budget(
                time_budget_sec,
                is_simulate=is_simulate,
                only_phase1=only_phase1)

            for idx in range(len(noisy_degree)):
                ele = noisy_degree[idx]
                all_lines_auc[f"noisy degree - {ele}"][run_id].append(_degree_auc[idx])

    # draw the graph
    draw_list = []
    for key, value in all_lines_auc.items():
        one_line = (
            budget_array, value, key
        )
        draw_list.append(one_line)
    plot_experiment(draw_list, "Noisy Degree")
    print(budget_array)

    """
    time_budget = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    noisy_degree =[0, 0.3, 0.5, 0.7, 1]
    
    noisy_degree =0, 
    sys_acc_m=[0.9793351  0.97965873 0.98016251 0.98029751 0.98035841 0.98054162
    0.98051063 0.98056461 0.9805699  0.98066832 0.98067426], 
    sys_acc_m_25=[0.97905167 0.97914614 0.98002308 0.98009476 0.98009476 0.98023211
    0.98042289 0.98042289 0.98042289 0.98052188 0.98052188]
    
    noisy_degree =0.3, 
    sys_acc_m=[0.9795417  0.97958833 0.98008253 0.98025126 0.98028262 0.98048641
    0.98041702 0.98061343 0.98057749 0.98061622 0.98069128], 
    sys_acc_m_25=[0.97919281 0.97908462 0.97984906 0.98004839 0.98009476 0.98023211
    0.98009476 0.98042289 0.98042289 0.98044764 0.98052188]
    
    noisy_degree =0.5, 
    sys_acc_m=[0.97936154 0.9794479  0.98001024 0.98012287 0.98025332 0.98042334
    0.98057988 0.98066708 0.98063632 0.98074862 0.98081903], 
    sys_acc_m_25=[0.97896401 0.97892876 0.97975342 0.97986233 0.98002308 0.98010087
    0.98030292 0.98042289 0.98030292 0.98042289 0.98075261]
    
    noisy_degree =0.7, 
    sys_acc_m=[0.97908527 0.97966749 0.97989573 0.98018679 0.98037999 0.98046169
    0.98049885 0.9805969  0.98071688 0.98068621 0.98075563], 
    sys_acc_m_25=[0.97866051 0.97914367 0.97952316 0.98002308 0.98009476 0.98016468
    0.98030292 0.98032052 0.98037335 0.98033628 0.98042289]
    
    noisy_degree =1, 
    sys_acc_m=[0.97804819 0.9786629  0.97928355 0.97933981 0.97945496 0.97969818
    0.97977445 0.97973729 0.9798775  0.97990203 0.97995017], 
    sys_acc_m_25=[0.97741743 0.9780354  0.97901241 0.97897443 0.9791354  0.9794121
    0.9795258  0.97945627 0.97965465 0.97958102 0.9796478 ]
    
    """



