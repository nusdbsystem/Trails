import calendar
import json
import os
import random
import sqlite3
import time
import traceback

import numpy as np
import torch
from exps.shared_args import parse_arguments


def initialize_logger(args):
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", f"{args.log_name}_{args.dataset}_wkid_{args.worker_id}_{ts}.log")
    os.environ.setdefault("base_dir", args.base_dir)


def set_random_seed():
    seed = 20
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def initialize_experiment(args):
    logger.info("running with params: :" + json.dumps(args.__dict__, indent=2))
    logger.info("cuda available = " + str(torch.cuda.is_available()))

    local_api = ImgScoreQueryApi(args.search_space, args.dataset)
    search_space_ins = init_search_space(args)
    return local_api, search_space_ins


def initialize_database(tfmem_smt_file):
    connection = sqlite3.connect(tfmem_smt_file)

    try:
        connection.execute(
            "CREATE TABLE simulateExp(run_num, model_explored, cur_arch_id, top200_model_list, current_x_time)")
        connection.execute("CREATE INDEX index_name on simulateExp (run_num, model_explored);")
    except:
        raise
    return connection


def run_experiment(num_run, num_arch, local_api, search_space_ins, connection, args):
    all_run_info = {}
    run_begin_time = time.time()
    for run_id in range(num_run):
        all_run_info.update(execute_single_run(run_id, num_arch, local_api, search_space_ins, connection, args))
        logger.info("run {} finished using {}".format(run_id, time.time() - run_begin_time))

    connection.commit()
    return all_run_info


def execute_single_run(run_id, num_arch, local_api, search_space_ins, connection, args):
    run_begin_time = time.time()
    strategy = RegularizedEASampler(search_space_ins,
                                    population_size=args.population_size,
                                    sample_size=args.sample_size)
    sampler = SampleController(strategy)

    total_ge_time = total_fit_time = total_compute_time = total_record_time = 0
    current_x_time = 0
    x_axis_time, arch_id_list, y_axis_top10_models = [], [], []

    i = 1
    try:
        while i <= num_arch:
            run_info = execute_single_architecture(i, sampler, local_api, run_id, current_x_time, connection, args)

            total_ge_time += run_info['ge_time']
            total_fit_time += run_info['fit_time']
            total_compute_time += run_info['compute_time']
            total_record_time += run_info['record_time']

            current_x_time += run_info['x_time']
            x_axis_time.append(current_x_time)

            arch_id_list.append(run_info['arch_id'])
            y_axis_top10_models.append(run_info['top_models'])

            i += 1

    except Exception as e:
        handle_exceptions(e)

    print_run_info(run_id, run_begin_time, total_ge_time, total_fit_time, total_compute_time, total_record_time)

    return {
        run_id: {"arch_id_list": arch_id_list, "y_axis_top10_models": y_axis_top10_models, "x_axis_time": x_axis_time}}


def execute_single_architecture(i, sampler, local_api, run_id, current_x_time, connection, args):
    begin_ge_model = time.time()
    arch_id, arch_micro = sampler.sample_next_arch()
    ge_time = time.time() - begin_ge_model

    begin_get_score = time.time()
    naswot_score = local_api.api_get_score(str(arch_id))[CommonVars.NAS_WOT]
    synflow_score = local_api.api_get_score(str(arch_id))[CommonVars.PRUNE_SYNFLOW]
    compute_time = time.time() - begin_get_score

    begin_fit = time.time()
    alg_score = {CommonVars.NAS_WOT: naswot_score, CommonVars.PRUNE_SYNFLOW: synflow_score}
    sampler.fit_sampler(arch_id, alg_score, simple_score_sum=True)
    fit_time = time.time() - begin_fit

    begin_record = time.time()
    x_time = guess_score_time(args.search_space, args.dataset)
    top_models = json.dumps(sampler.get_current_top_k_models(400))

    insert_str = f"INSERT INTO simulateExp VALUES ({run_id}, {i}, {arch_id}, '{top_models}', {round(current_x_time + x_time, 4)})"
    connection.execute(insert_str)

    record_time = time.time() - begin_record

    return {"arch_id": arch_id, "top_models": top_models, "x_time": x_time,
            "ge_time": ge_time, "fit_time": fit_time, "compute_time": compute_time, "record_time": record_time}


def handle_exceptions(e):
    print(traceback.format_exc())
    logger.info("========================================================================")
    logger.error(traceback.format_exc())
    logger.error("error: " + str(e))
    logger.info("========================================================================")
    exit(1)


def print_run_info(run_id, run_begin_time, total_ge_time, total_fit_time, total_compute_time, total_record_time):
    print("run {} finished using {}".format(run_id, time.time() - run_begin_time))
    print(f"total_ge_time = {total_ge_time}, total_fit_time = {total_fit_time}, total_compute_time = "
          f"{total_compute_time}, total_record_time = {total_record_time}")


if __name__ == '__main__':
    args = parse_arguments()
    initialize_logger(args)

    from src.common.constant import CommonVars
    from src.controller.controler import SampleController
    from src.controller.sampler_ea.regularized_ea import RegularizedEASampler
    from src.logger import logger
    from src.query_api.query_api_img import guess_score_time, ImgScoreQueryApi
    from src.search_space.init_search_space import init_search_space

    set_random_seed()
    local_api, search_space_ins = initialize_experiment(args)
    tfmem_smt_file = \
        f"{args.result_dir}/TFMEM_{args.search_space}_{args.dataset}_100run_8k_models_sumed_score"
    connection = initialize_database(tfmem_smt_file)
    all_run_info = run_experiment(
        num_run=100,
        num_arch=8000,
        local_api=local_api,
        search_space_ins=search_space_ins,
        connection=connection,
        args=args)
