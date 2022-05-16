import calendar
import os
import random
import time
from exps.shared_args import parse_arguments
import torch
from src.tools.res_measure import print_cpu_gpu_usage
from multiprocessing import Pool
from torch import nn
import argparse
from typing import List

args = parse_arguments()

random.seed(80)
# set the log name
gmt = time.gmtime()
ts = calendar.timegm(gmt)
os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
os.environ.setdefault("base_dir", args.base_dir)

from src.common.structure import ModelAcquireData
from src.controller.sampler_all.seq_sampler import SequenceSampler
from src.search_space.init_search_space import init_search_space
from src.dataset_utils.structure_data_loader import libsvm_dataloader
from src.tools.io_tools import write_json, read_json
from src.dataset_utils import dataset
from src.common.constant import Config
from src.eva_engine.phase1.concurrent_evaluator import ConcurrentP1Evaluator
from src.search_space.init_search_space import SpaceWrapper


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


# define func here for import
def evaluate_model(
        arch: nn.Module,
        args: argparse,
        search_space_ins: SpaceWrapper,
        _evaluator: ConcurrentP1Evaluator,
        explored_n: int,
        result: List):
    arch_id, arch_micro = arch
    if arch_id in result:
        return None
    if explored_n > args.models_explore:
        return None

    model_encoding = search_space_ins.serialize_model_encoding(arch_micro)
    model_acquire_data = ModelAcquireData(model_id=arch_id,
                                          model_encoding=model_encoding,
                                          is_last=False)
    data_str = model_acquire_data.serialize_model()
    model_score = _evaluator.p1_evaluate(data_str)

    if explored_n % 50 == 0 and _evaluator.if_cuda_avaiable():
        begin = time.time()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return arch_id, model_score


if __name__ == "__main__":

    # main proces here
    output_file = f"{args.result_dir}/score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.concurrency}.json"
    time_output_file = f"{args.result_dir}/time_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.concurrency}.json"
    res_output_file = f"{args.result_dir}/resource_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.concurrency}.json"

    # start the resource monitor
    stop_event, thread = print_cpu_gpu_usage(interval=0.5, output_file=res_output_file)

    search_space_ins = init_search_space(args)
    train_loader, val_loader, test_loader, class_num = generate_data_loader()
    _evaluator = ConcurrentP1Evaluator(device=args.device,
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

    overall_latency = []
    begin = time.time()
    # concurrent model evaluation
    with Pool(processes=args.concurrency) as pool:
        archs_to_evaluate = [sampler.sample_next_arch() for _ in range(args.models_explore)]
        total_to_evaluate = len(archs_to_evaluate)
        for i, res in enumerate(pool.starmap(
                evaluate_model,
                [(arch, args, search_space_ins, _evaluator, explored_n, result)
                 for arch in archs_to_evaluate])):
            if i % 100 == 0:
                print(f"Progress: {i}/{total_to_evaluate}")
            if res is not None:
                arch_id, model_score = res
                result[arch_id] = model_score
                explored_n += 1

    # everything is done, record the time usage.
    if _evaluator.if_cuda_avaiable():
        torch.cuda.synchronize()

    end = time.time()
    overall_latency.append(end - begin)

    write_json(output_file, result)
    # compute time
    write_json(time_output_file, overall_latency)

    # Then, at the end of your program, you can stop the thread:
    print("Done, time sleep for 10 seconds")
    # wait the resource montor flush
    time.sleep(10)
    stop_event.set()
    thread.join()
