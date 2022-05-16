import calendar
import os
import time
from exps.shared_args import parse_arguments

args = parse_arguments()

# set the log name
gmt = time.gmtime()
ts = calendar.timegm(gmt)
os.environ.setdefault("log_logger_folder_name", f"bm_filter_phase")
os.environ.setdefault("log_file_name", f"bm_filter_{args.dataset}_{args.device}" + "_" + str(ts) + ".log")
os.environ.setdefault("base_dir", args.base_dir)

from src.eva_engine.run_ms import RunModelSelection
from src.common.constant import Config, CommonVars
from src.query_api.query_api_img import guess_score_time, ImgScoreQueryApi
from src.query_api.query_api_mlp import GTMLP


def debug_args(args, dataset):
    args.dataset = dataset
    args.base_dir = "../exp_data/"
    args.is_simulate = True
    args.log_folder = "log_ku_tradeoff"

    if dataset == Config.Frappe:
        args.search_space = "mlp_sp"
        args.epoch = 13
        args.hidden_choice_len = 20
        y_label = "FRAPPE"
        gtapi = GTMLP(dataset)
        score_time = gtapi.get_score_one_model_time("cpu")

    if dataset == Config.UCIDataset:
        args.search_space = "mlp_sp"
        args.epoch = 14
        args.hidden_choice_len = 20
        y_label = "DIABETES"
        gtapi = GTMLP(dataset)
        score_time = gtapi.get_score_one_model_time("cpu")

    if dataset == Config.Criteo:
        args.search_space = "mlp_sp"
        args.epoch = 9
        args.hidden_choice_len = 10
        y_label = "CRITEO"
        gtapi = GTMLP(dataset)
        score_time = gtapi.get_score_one_model_time("cpu")

    if dataset == Config.c10:
        args.search_space = "nasbench201"
        args.epoch = 200
        y_label = "C10"
        score_time = guess_score_time(Config.NB201, Config.c10)
        gtapi = ImgScoreQueryApi(Config.NB201, Config.c10)

    if dataset == Config.c100:
        args.search_space = "nasbench201"
        args.epoch = 180
        y_label = "C100"
        score_time = guess_score_time(Config.NB201, Config.c100)
        gtapi = ImgScoreQueryApi(Config.NB201, Config.c100)

    if dataset == Config.imgNet:
        args.search_space = "nasbench201"
        args.epoch = 200
        y_label = "IN-16"
        score_time = guess_score_time(Config.NB201, Config.imgNet)
        gtapi = ImgScoreQueryApi(Config.NB201, Config.imgNet)

    return y_label, score_time, gtapi


def update_alg_name(name):
    if name == CommonVars.PRUNE_SYNFLOW:
        return "SynFlow"
    if name == CommonVars.JACFLOW:
        return "JacFlow"
    if name == CommonVars.PRUNE_SNIP:
        return "SNIP"

    return name


def get_top_models(space_name, api, score_alg_name, K):
    if space_name == Config.MLPSP:
        if score_alg_name == CommonVars.JACFLOW:
            # larget to small
            sorted_items = sorted(api.mlp_global_rank.items(), key=lambda item: item[1]['nas_wot_synflow'],
                                  reverse=True)
            top_k_items = [item[0] for item in sorted_items[:K]]
            return top_k_items
        else:
            sorted_items = sorted(api.mlp_score.items(), key=lambda item: float(item[1][score_alg_name]), reverse=True)
            top_k_items = [item[0] for item in sorted_items[:K]]
            return top_k_items
    else:
        if score_alg_name == CommonVars.JACFLOW:
            # larget to small
            sorted_items = sorted(api.global_rank.items(), key=lambda item: item[1]['nas_wot_synflow'], reverse=True)
            top_k_items = [item[0] for item in sorted_items[:K]]
            return top_k_items
        else:
            sorted_items = sorted(api.data.items(), key=lambda item: float(item[1].get(score_alg_name, 0)),
                                  reverse=True)
            top_k_items = [item[0] for item in sorted_items[:K]]
            return top_k_items


if __name__ == "__main__":

    # this is for ploting the graph, 10000
    for n in [9500]:

        dataset_res = {}
        for dataset in [Config.c100]:
        # for dataset in [Config.c100, Config.c10, Config.Frappe, Config.UCIDataset, Config.Criteo, Config.imgNet]:
            y_label, score_time, gtapi = debug_args(args, dataset)
            args.epoch = int(input())
            rms = RunModelSelection(args.search_space, args, is_simulate=True)

            # this is for ploting the graph
            dataset_res[y_label] = []

            print("============" * 20)
            print(dataset)

            for training_free_alg in [CommonVars.JACFLOW, CommonVars.PRUNE_SYNFLOW, CommonVars.PRUNE_SNIP]:
                for budget_aware_alg in [Config.SUCCHALF, Config.SUCCREJCT, Config.UNIFORM]:
                    k_models = get_top_models(args.search_space, gtapi, training_free_alg, int(n / 100))
                    best_arch, best_arch_performance, _, total_time_usage = rms.refinement_phase(
                        U=1, k_models=k_models,
                        alg_name=budget_aware_alg)

                    total_time = total_time_usage + score_time * len(k_models)

                    # print(
                    #     f"Running task budget_aware_alg = {budget_aware_alg}, metrics = {training_free_alg}, "
                    #     f"0.25-0.75 = {q_25_y}, {mean_y}, {q_75_y},"
                    #     f" total_time_usage={mean_time}")

                    dataset_res[y_label].append(
                        [update_alg_name(training_free_alg) + " + " + budget_aware_alg, best_arch_performance,
                         total_time]
                    )
        print("=========================>", n, dataset_res)
