import calendar
import json
import os
import random
import time
from exps.shared_args import parse_arguments
from src.common.constant import Config, CommonVars


def debug_args(args):
    args.tfmem = "express_flow"
    args.search_space = "mlp_sp"
    args.epoch = 14
    args.dataset = "mfeat217"
    args.base_dir = "../exp_data/"
    args.is_simulate = True
    args.log_folder = "log_ku_tradeoff"

    args.num_layers = 4
    args.hidden_choice_len = 20
    args.batch_size = 1024
    args.nfeat = 217
    args.nfield = 2000
    args.num_labels = 2
    args.only_phase1 = False
    # this is from paper KNAS
    args.models_explore = 100000


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
    args = parse_arguments()

    debug_args(args)

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.common.structure import ModelAcquireData
    from src.controller.sampler_all.seq_sampler import SequenceSampler
    from src.eva_engine.phase1.evaluator import P1Evaluator
    from src.logger import logger
    from src.search_space.init_search_space import init_search_space
    from src.tools.io_tools import write_json, read_json

    search_space_ins = init_search_space(args)

    _evaluator = P1Evaluator(device=args.device,
                             num_label=args.num_labels,
                             dataset_name=args.dataset,
                             search_space_ins=search_space_ins,
                             train_loader=None,
                             is_simulate=False,
                             metrics=args.tfmem,
                             enable_cache=args.embedding_cache_filtering)

    sampler = SequenceSampler(search_space_ins)

    output_file = f"{args.result_dir}/score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}.json"
    result = read_json(output_file)
    sorted_keys = sorted(result, key=lambda k: result[k]['express_flow'], reverse=True)[:10]
    print(sorted_keys)

    explored_n = 0
    print(f"begin to score all, currently we already explored {len(result.keys())}")
    logger.info(f"begin to score all, currently we already explored {len(result.keys())}")
    while True:
        arch_id, arch_micro = sampler.sample_next_arch()
        if arch_id is None:
            logger.info("Stop exploring, meet None arch id")
            break
        if arch_id in result:
            continue
        if args.models_explore != -1 and explored_n > args.models_explore:
            logger.info(f"Stop exploring, {explored_n} > {args.models_explore}")
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
        # print(f" {datetime.now()} finish arch = {arch_id}, model_score = {model_score}")

        if explored_n < 10:
            print("3. [trails] Phase 1: filter phase explored " + str(explored_n)
                  + "Total explored " + str(len(result)) +
                  " model, model_id = " + str(arch_id) +
                  " model_scores = " + json.dumps(model_score))
            logger.info("3. [trails] Phase 1: filter phase explored " + str(explored_n)
                        + "Total explored " + str(len(result)) +
                        " model, model_id = " + str(arch_id) +
                        " model_scores = " + json.dumps(model_score))
        if explored_n % 1000 == 0:
            # print_memory_usg()
            # _evaluator.force_gc()
            print("3. [trails] Phase 1: filter phase explored " + str(explored_n)
                  + "Total explored " + str(len(result)) +
                  " model, model_id = " + str(arch_id) +
                  " model_scores = " + json.dumps(model_score))
            logger.info("3. [trails] Phase 1: filter phase explored " + str(explored_n)
                        + "Total explored " + str(len(result)) +
                        " model, model_id = " + str(arch_id) +
                        " model_scores = " + json.dumps(model_score))
        if explored_n % 1000 == 0:
            # print_memory_usg()
            # _evaluator.force_gc()
            logger.info("3. [trails] Phase 1: filter phase explored " + str(explored_n) +
                        " model, model_id = " + str(arch_id) +
                        " model_scores = " + json.dumps(model_score))
            write_json(output_file, result)
    write_json(output_file, result)
