import calendar
import os
import time
from exps.shared_args import parse_arguments, seed_everything
import random

args = parse_arguments()

# set the log name
gmt = time.gmtime()
ts = calendar.timegm(gmt)
os.environ.setdefault("log_logger_folder_name", f"bm_filter_phase")
os.environ.setdefault("log_file_name", f"bm_filter_{args.dataset}_{args.device}" + "_" + str(ts) + ".log")
os.environ.setdefault("base_dir", args.base_dir)

from src.eva_engine.phase1.evaluator import P1Evaluator
from src.search_space.init_search_space import init_search_space
from src.dataset_utils.structure_data_loader import libsvm_dataloader
from src.tools.io_tools import write_json, read_json
from src.dataset_utils import dataset
from src.common.constant import Config
from src.query_api.query_api_mlp import GTMLP


def debug_args(args):
    if args.dataset == Config.Frappe:
        args.tfmem = "knas"
        args.search_space = "mlp_sp"
        args.epoch = 14
        args.dataset = "frappe"
        args.base_dir = "../exp_data/"
        args.is_simulate = True
        args.log_folder = "log_ku_tradeoff"

        args.num_layers = 4
        args.hidden_choice_len = 20
        args.batch_size = 1024
        args.nfeat = 5500
        args.nfield = 10
        args.num_labels = 2
        args.only_phase1 = False

        # this is from paper KNAS
        args.models_explore = 100
        score_time = 22.58024501800537
        return score_time

    if args.dataset == Config.Criteo:
        args.tfmem = "knas"
        args.search_space = "mlp_sp"
        args.epoch = 9
        args.dataset = "criteo"
        args.base_dir = "../exp_data/"
        args.is_simulate = True
        args.log_folder = "log_ku_tradeoff"

        args.num_layers = 4
        args.hidden_choice_len = 10
        args.batch_size = 1024
        args.nfeat = 2100000
        args.nfield = 39
        args.num_labels = 2
        args.only_phase1 = False

        # this is from paper KNAS
        args.models_explore = 100
        score_time = 25.58024501800537
        return score_time

    if args.dataset == Config.UCIDataset:
        args.tfmem = "knas"
        args.search_space = "mlp_sp"
        args.epoch = 1
        args.dataset = "uci_diabetes"
        args.base_dir = "../exp_data/"
        args.is_simulate = True
        args.log_folder = "log_ku_tradeoff"

        args.num_layers = 4
        args.hidden_choice_len = 20
        args.batch_size = 1024
        args.nfeat = 369
        args.nfield = 43
        args.num_labels = 2
        args.only_phase1 = False

        # this is from paper KNAS
        args.models_explore = 100
        score_time = 27.58024501800537
        return score_time


if __name__ == "__main__":

    output_file = f"{args.result_dir}/knas_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"
    time_output_file = f"{args.result_dir}/time_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"

    score_time = debug_args(args)
    gtapi = GTMLP(args.dataset)

    explored_n = 0
    score_his = read_json(output_file)

    all_run_res = []
    for run in range(2000):
        selected_items = random.sample(score_his.items(), 100)
        selected_dict = dict(selected_items)

        sorted_models = sorted(selected_dict.items(), key=lambda item: item[1]["knas"])
        # Step 2: Create a new dictionary from the sorted items. We use a dictionary comprehension here.
        models_all = [model_id for model_id, v in sorted_models]
        print(models_all)
        # we set K=5 following the paper KNAS
        models = models_all[-5:]

        train_time = 0
        global_best = 0
        for ele in models:
            auc, _time = gtapi.get_valid_auc(ele, args.epoch)
            train_time += _time
            if auc > global_best:
                global_best = auc
        print(score_time, train_time)
        print("Time-usage", train_time + score_time, "Auc", global_best)
        all_run_res.append(global_best)
    print("Max Auc", max(all_run_res))
