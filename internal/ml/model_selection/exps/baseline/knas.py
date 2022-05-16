import calendar
import os
import time
from exps.shared_args import parse_arguments, seed_everything

args = parse_arguments()

# set the log name
gmt = time.gmtime()
ts = calendar.timegm(gmt)
os.environ.setdefault("log_logger_folder_name", f"bm_filter_phase")
os.environ.setdefault("log_file_name", f"bm_filter_{args.dataset}_{args.device}" + "_" + str(ts) + ".log")
os.environ.setdefault("base_dir", args.base_dir)

from src.common.structure import ModelAcquireData
from src.controller.sampler_rand.random_sample import RandomSampler
from src.eva_engine.phase1.evaluator import P1Evaluator
from src.search_space.init_search_space import init_search_space
from src.dataset_utils.structure_data_loader import libsvm_dataloader
from src.tools.io_tools import write_json, read_json
from src.dataset_utils import dataset
from src.common.constant import Config
from src.query_api.query_api_mlp import GTMLP


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


if __name__ == "__main__":

    output_file = f"{args.result_dir}/knas_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"
    time_output_file = f"{args.result_dir}/time_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"

    debug_args(args)
    gtapi = GTMLP(args.dataset)

    search_space_ins = init_search_space(args)
    train_loader, val_loader, test_loader, class_num = generate_data_loader()
    _evaluator = P1Evaluator(device=args.device,
                             num_label=args.num_labels,
                             dataset_name=args.dataset,
                             search_space_ins=search_space_ins,
                             train_loader=train_loader,
                             is_simulate=False,
                             metrics=args.tfmem,
                             enable_cache=args.embedding_cache_filtering)
    explored_n = 0
    score_his = read_json(output_file)

    sampler = RandomSampler(search_space_ins)
    while True:
        arch_id, arch_micro = sampler.sample_next_arch()
        if arch_id is None:
            break
        if arch_id in score_his:
            continue
        if explored_n > args.models_explore:
            break
        # run the model selection
        model_encoding = search_space_ins.serialize_model_encoding(arch_micro)
        model_acquire_data = ModelAcquireData(model_id=arch_id,
                                              model_encoding=model_encoding,
                                              is_last=False)
        data_str = model_acquire_data.serialize_model()
        model_score = _evaluator.p1_evaluate(data_str)
        explored_n += 1
        score_his[arch_id] = model_score
        if explored_n % 50 == 0:
            print(f"Evaluate {explored_n} models")

    write_json(output_file, score_his)
    write_json(time_output_file, score_his)

    score_time = sum(_evaluator.time_usage["track_compute"])
    # now train each model,

    sorted_models = sorted(score_his.items(), key=lambda item: item[1]["knas"])
    # Step 2: Create a new dictionary from the sorted items. We use a dictionary comprehension here.
    models_all = [model_id for model_id, v in sorted_models]
    # we set K=5 following the paper KNAS
    models = models_all[-5:]

    train_time = 0
    global_best = 0
    for ele in models:
        auc, _time = gtapi.get_valid_auc(ele, args.epoch)
        train_time += _time
        if auc > global_best:
            global_best = auc
    print(models_all)
    print(score_time, train_time)
    print("Time-usage", train_time + score_time, "Auc", global_best)
