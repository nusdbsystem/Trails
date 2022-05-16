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
from src.tools.io_tools import write_json, read_json


def debug_args(args, dataset):
    args.dataset = dataset
    args.base_dir = "../exp_data/"
    args.is_simulate = True
    args.log_folder = "log_ku_tradeoff"

    if dataset == Config.Frappe:
        args.search_space = "mlp_sp"
        args.epoch = 13
        args.hidden_choice_len = 20

    if dataset == Config.UCIDataset:
        args.search_space = "mlp_sp"
        args.epoch = 0
        args.hidden_choice_len = 20

    if dataset == Config.Criteo:
        args.search_space = "mlp_sp"
        args.epoch = 9
        args.hidden_choice_len = 10

    if dataset == Config.c10:
        args.search_space = "nasbench201"
        args.epoch = 200

    if dataset == Config.c100:
        args.search_space = "nasbench201"
        args.epoch = 200

    if dataset == Config.imgNet:
        args.search_space = "nasbench201"
        args.epoch = 200


if __name__ == "__main__":
    # (100 run + SUCCHALF), random seed 2201
    output_file = f"{args.result_dir}/score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"
    time_output_file = f"{args.result_dir}/time_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"

    for n in [10000]:
        k = int(n / 100)

        # turn on this to run the image
        for dataset in [Config.Frappe, Config.UCIDataset, Config.Criteo, Config.c10, Config.c100, Config.imgNet]:
            debug_args(args, dataset)
            metrics_list = [CommonVars.PRUNE_SYNFLOW, CommonVars.NAS_WOT, CommonVars.PRUNE_SNIP, CommonVars.JACFLOW]
            result = {}
            for ele in metrics_list:
                result[ele] = []
            for run in range(100):
                for training_free_alg in metrics_list:
                    args.tfmem = training_free_alg

                    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
                    k_models, _, _, _ = rms.filtering_phase(N=n, K=k)
                    result[training_free_alg].append(k_models)
            write_json(f"./internal/ml/model_selection/exp_result/combine_train_free_based_{dataset}_n_{n}.josn", result)
