import random
import time

from src.tools.io_tools import read_json
from exps.shared_args import parse_arguments
import os
from typing import Set, List, Dict

args = parse_arguments()
os.environ.setdefault("base_dir", args.base_dir)

from src.search_space.init_search_space import init_search_space
from src.controller.sampler_ea.regularized_ea import AsyncRegularizedEASampler
from src.controller.controler import SampleController
import queue
import threading


class DictToObject(object):
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


def random_generate_models():
    arch_scores = read_json(
        "./internal/ml/model_selection/exp_result/exp_filtering_scalability/"
        "score_mlp_sp_criteo_batch_size_64_cpu_jacflow.json")
    result = {}
    arch_infos = read_json(
        "./internal/ml/model_selection/exp_result/exp_filtering_scalability/"
        "time_score_mlp_sp_criteo_batch_size_32_cpu_jacflow.json")
    result['model_id'] = arch_infos["model_id"]
    result['time_usage'] = [sum(values) for values in
                            zip(arch_infos['track_io_model_init'],
                                arch_infos['track_io_data_retrievel'],
                                arch_infos['track_io_data_preprocess'],
                                arch_infos['track_compute'])]

    res = {}
    for index in range(len(result['model_id'])):
        _model = result['model_id'][index]
        res[_model] = {
            "time": result['time_usage'][index],
            "score": arch_scores[_model],
        }
    # print(len(res))
    return res


eta = 3

score_queue = queue.Queue()
arch_id_queue = queue.Queue()
all_worker_res = queue.Queue()


# A function for threads to insert data into the queue.
def inserter(worker_num):
    # here we only measure on criteo
    args.hidden_choice_len = 10
    args.nfield = 39
    args.nfeat = 2100000
    args.nemb = 10
    args.num_layers = 4
    args.num_labels = 2
    args.search_space = "mlp_sp"
    search_space_ins = init_search_space(args)
    strategy = AsyncRegularizedEASampler(
        search_space_ins,
        population_size=10,
        sample_size=3)

    sampler = SampleController(strategy)

    # init for all workers
    for _ in range(worker_num):
        arch_id, arch_micro = sampler.sample_next_arch()
        arch_id_queue.put([arch_id, arch_micro])

    explored_n = 0
    while explored_n < 9900:
        arch_id, arch_micro = sampler.sample_next_arch()
        arch_id_queue.put([arch_id, arch_micro])
        # print(f"[coordinator]: explore {explored_n} models, getting from score queue...")
        item = score_queue.get()
        sampler.fit_sampler(arch_id=item[0], arch_micro=item[1], alg_score=item[2], is_sync=False)
        explored_n += 1

    # print(f"[coordinator]: Done")
    for _ in range(worker_num):
        arch_id_queue.put(["stop", "stop"])


def evaluate(worker_index, my_obj_m: Dict):
    cur_time = 0
    worker_tracker = {worker_index: {
        "time_usage": [],
        "score_get": [],
    }}
    times = []
    scores = []
    while True:
        # print(f"[{worker_index}]: Getting arch id from arch_id_queue....")
        item = arch_id_queue.get()
        arch_id, arch_micro = item[0], item[1]
        if arch_id == "stop":
            # print(f"[{worker_index}]: Stop")
            break
        cur_time += my_obj_m[arch_id]["time"]
        # simulate the latency of each worker
        time.sleep(my_obj_m[arch_id]["time"] / 5000)
        times.append(cur_time)
        if len(scores) == 0 or my_obj_m[arch_id]["score"]["jacflow"] > scores[-1]:
            scores.append(my_obj_m[arch_id]["score"]["jacflow"])
        else:
            scores.append(scores[-1])
        score_queue.put([arch_id, arch_micro, my_obj_m[arch_id]["score"]])

    # record result
    # print(f"[{worker_index}]: Return result to the all_worker_res")
    worker_tracker[worker_index]["time_usage"] = times
    worker_tracker[worker_index]["score_get"] = scores
    all_worker_res.put(worker_tracker)


def worker_process(worker_num, my_obj_m: Dict):
    threads = []
    for i in range(worker_num):
        thread = threading.Thread(target=evaluate, args=(f"wk_{i}", my_obj_m))
        thread.start()
        threads.append(thread)
    return threads


def distributed_filtering(worker_num, my_obj_m: Dict):
    # print("Run workers")
    threads = worker_process(worker_num, my_obj_m)
    # print("Run inserter")
    inserter(worker_num)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    final_res = {}
    while not all_worker_res.empty():
        result = all_worker_res.get()
        final_res.update(result)

    # here is to debug
    # draw_graph_worker_level(final_res)
    return gather_worker_res(final_res)


def gather_worker_res(final_res):
    time_arrays = []
    auc_arrays = []
    for key, value in final_res.items():
        time_arrays.append(value["time_usage"])
        auc_arrays.append(value["score_get"])

    # This function finds the appropriate AUC value for an experiment at a given time.
    def get_auc_at_time(time_array, auc_array, query_time):
        # If the experiment hasn't started at the query time, we can't use its AUC.
        if query_time < time_array[0]:
            return -np.inf  # Represents an invalid AUC that won't be considered for the maximum.

        # Find the latest AUC measurement before the query time.
        for i in range(len(time_array)):
            if time_array[i] > query_time:
                return auc_array[i - 1] if i > 0 else -np.inf  # If no previous, return invalid AUC.

        # If we've gone through the whole array, the query time is after the last measurement.
        return auc_array[-1]  # The last known AUC value.

    # Set up your standard time scale.
    min_time = min(np.min(times) for times in time_arrays)
    max_time = max(np.max(times) for times in time_arrays)
    standard_time = np.linspace(min_time, max_time, num=1000)  # You can decide the appropriate number.

    max_auc_values = []
    for std_time in standard_time:
        # For this time point, we'll collect the relevant AUC from each experiment and find the max.
        auc_values_at_time = [get_auc_at_time(time, auc, std_time) for time, auc in zip(time_arrays, auc_arrays)]
        max_auc = max(auc_values_at_time)  # Max AUC value among all experiments at this time point.
        max_auc_values.append(max_auc)

    return list(standard_time), max_auc_values


# Define the formatter function
def thousands_formatter(x, pos):
    'The two args are the value and tick position'
    if x >= 1000:
        return '{}k'.format(int(x * 1e-3))  # No decimal places needed
    return str(x)


import matplotlib
from matplotlib.ticker import FuncFormatter

matplotlib.use('Agg')  # Use this line before importing pyplot or using any other Matplotlib functionality.
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def draw_graph(super_multi_run_info, phase, x_label):
    mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]
    line_shape_list = ['-.', '--', ':', '-', ]
    fig2, ax = plt.subplots(figsize=(6.5, 4))

    i = 0
    for _worker_num, info in super_multi_run_info.items():
        multi_run_time = info[0]
        multi_run_score = info[1]
        multi_run_time = np.array(multi_run_time)
        multi_run_score = np.array(multi_run_score)

        y_h = np.quantile(multi_run_score, .75, axis=0)
        y_m = np.quantile(multi_run_score, .5, axis=0)
        y_l = np.quantile(multi_run_score, .25, axis=0)

        x_h = np.quantile(multi_run_time, .75, axis=0)
        x_m = np.quantile(multi_run_time, .5, axis=0)
        x_l = np.quantile(multi_run_time, .25, axis=0)

        for index in range(len(np.log(y_m))):
            if np.log(y_m)[index] == 19.57964927397326:
                print(_worker_num, x_m[index])
                break

        ax.fill_between(x_m, np.log(y_l), np.log(y_h), alpha=0.1)
        ax.plot(x_m, np.log(y_m),
                line_shape_list[i],
                label=_worker_num, linewidth=4, markersize=40)

        i += 1

    ax.grid()
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(f"Normalized \n JacFlow Score", fontsize=20)
    formatter = FuncFormatter(thousands_formatter)
    ax.yaxis.set_major_formatter(formatter)
    # 'linear', 'log', 'symlog', 'asinh', 'logit', 'function', 'functionlog'
    # ax.set_xscale("log")
    # ax.set_xticks([1, 2, 4, 8])  # Specify which ticks you want
    plt.ylim(18.4, None)
    plt.xlim(None, 120)

    ax.legend(fontsize=15, ncol=1, loc='lower right')
    ax.tick_params(axis='both', labelsize=20)
    # print(f"./scale_{phase}.pdf")
    fig2.savefig(f"./scale_{phase.lower()}2.pdf", bbox_inches='tight')


def draw_graph_one_run(result):
    x, y = result[0], result[1]
    fig2, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, np.log(y), linewidth=4, markersize=4)
    ax.grid()
    ax.set_ylabel(f"Normalized \n JacFlow Score", fontsize=20)
    formatter = FuncFormatter(thousands_formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(fontsize=12, ncol=1, loc='upper right')
    ax.tick_params(axis='both', labelsize=20)
    fig2.savefig(f"./scale_debug_worker.pdf", bbox_inches='tight')


def draw_graph_worker_level(result):
    fig2, ax = plt.subplots(figsize=(6.5, 4))
    for key, value in result.items():
        ax.plot(value["time_usage"], np.log(value["score_get"]), label=key, linewidth=4, markersize=4)
    ax.grid()
    ax.set_ylabel(f"Normalized \n JacFlow Score", fontsize=20)
    formatter = FuncFormatter(thousands_formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.legend(fontsize=12, ncol=1, loc='upper right')
    ax.tick_params(axis='both', labelsize=20)
    fig2.savefig(f"./scale_filtering_one_run.pdf", bbox_inches='tight')


def seed_everything(seed=11):
    ''' [reference] https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335 '''
    random.seed(seed)


if __name__ == "__main__":

    from src.tools.io_tools import read_json

    super_multi_run_info = read_json("./internal/ml/model_selection/exp_result/filter_scale.json")
    if super_multi_run_info != {}:
        draw_graph(super_multi_run_info, "filtering", "Total Time of the Filtering Phase (s)")
        exit(0)

    from src.tools.io_tools import write_json

    seed_everything()

    # 1. compute and draw the filtering phase
    n_models = random_generate_models()

    super_multi_run_info = {}

    worker_num = [1, 2, 4, 8]
    for _worker_num in worker_num:

        multi_run_time = []
        multi_run_score = []
        for i in range(20):
            print(f"_worker_num={_worker_num}, i = {i}")
            _times, _scores = distributed_filtering(_worker_num, n_models)
            multi_run_time.append(_times)
            multi_run_score.append(_scores)

        if _worker_num == 1:
            _namne = f"{_worker_num} CPU"
        else:
            _namne = f"{_worker_num} CPUs"

        super_multi_run_info[_namne] = [multi_run_time, multi_run_score]

    write_json("./internal/ml/model_selection/exp_result/filter_scale.json", super_multi_run_info)
    draw_graph(super_multi_run_info, "filtering", "Total Time of the Filtering Phase (s)")
