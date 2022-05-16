import calendar
import os
import random
import time
from exps.shared_args import parse_arguments
import torch
from src.tools.res_measure import print_cpu_gpu_usage
import gc
from src.tools.res_measure import print_memory_usage
args = parse_arguments()


# set the log name
gmt = time.gmtime()
ts = calendar.timegm(gmt)
os.environ.setdefault("log_logger_folder_name", f"bm_filter_phase")
os.environ.setdefault("log_file_name", f"bm_filter_{args.dataset}_{args.device}" + "_" + str(ts) + ".log")
os.environ.setdefault("base_dir", args.base_dir)

from src.logger import logger
from src.common.structure import ModelAcquireData
from src.controller.sampler_all.seq_sampler import SequenceSampler
from src.eva_engine.phase1.evaluator import P1Evaluator
from src.search_space.init_search_space import init_search_space
from src.dataset_utils.structure_data_loader import libsvm_dataloader
from src.tools.io_tools import write_json, read_json
from src.dataset_utils import dataset
from src.common.constant import Config


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


if __name__ == "__main__":

    output_file = f"{args.result_dir}/score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"
    time_output_file = f"{args.result_dir}/time_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"
    res_output_file = f"{args.result_dir}/resource_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"

    # start the resource monitor
    stop_event, thread = print_cpu_gpu_usage(interval=0.5, output_file=res_output_file)

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
    sampler = SequenceSampler(search_space_ins)
    explored_n = 0
    result = read_json(output_file)
    print(f"begin to score all, currently we already explored {len(result.keys())}")
    logger.info(f"begin to score all, currently we already explored {len(result.keys())}")

    while True:
        arch_id, arch_micro = sampler.sample_next_arch()
        if arch_id is None:
            break
        if arch_id in result:
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
        result[arch_id] = model_score
        if explored_n % 50 == 0:
            logger.info(f"Evaluate {explored_n} models")
            print(f"Evaluate {explored_n} models")

    if _evaluator.if_cuda_avaiable():
        torch.cuda.synchronize()

    # the first two are used for warming up
    _evaluator.time_usage["io_latency"] = \
        sum(_evaluator.time_usage["track_io_model_load"][2:]) + \
        sum(_evaluator.time_usage["track_io_model_init"][2:]) + \
        sum(_evaluator.time_usage["track_io_res_load"][2:])

    _evaluator.time_usage["compute_latency"] = sum(_evaluator.time_usage["track_compute"][2:])
    _evaluator.time_usage["latency"] = _evaluator.time_usage["io_latency"] + _evaluator.time_usage["compute_latency"]

    _evaluator.time_usage["avg_compute_latency"] = \
        _evaluator.time_usage["compute_latency"] \
        / len(_evaluator.time_usage["track_compute"][2:])

    write_json(output_file, result)
    # compute time
    write_json(time_output_file, _evaluator.time_usage)

    # Then, at the end of your program, you can stop the thread:
    print("Done, time sleep for 10 seconds")
    # wait the resource montor flush
    time.sleep(10)
    stop_event.set()
    thread.join()
