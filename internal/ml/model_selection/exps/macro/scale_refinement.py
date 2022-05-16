import os

os.environ.setdefault("base_dir", "../exp_data/")

from src.common.constant import Config
from src.query_api.query_api_mlp import GTMLP
import math
from src.search_space.mlp_api.space import DEFAULT_LAYER_CHOICES_10


# Define the formatter function
def thousands_formatter(x, pos):
    'The two args are the value and tick position'
    if x >= 1000:
        return '{}k'.format(int(x * 1e-3))  # No decimal places needed
    return str(int(x))


import matplotlib
from matplotlib.ticker import FuncFormatter

matplotlib.use('Agg')  # Use this line before importing pyplot or using any other Matplotlib functionality.
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import warnings

# 忽略DeprecationWarning警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

gtapi = GTMLP(Config.Criteo)


def random_generate_models(num_models):
    res = []
    while True:
        if len(res) == num_models:
            break
        arch_encod = []
        for _ in range(4):
            layer_size = str(random.choice(DEFAULT_LAYER_CHOICES_10))
            arch_encod.append(layer_size)
        if "-".join(arch_encod) in res:
            continue
        else:
            res.append("-".join(arch_encod))
    return res


eta = 3

import random


def divide_list(lst, m):
    # Shuffle the list in place
    random.shuffle(lst)

    # Calculate the size of each part and the remainder
    part_size = len(lst) // m
    remainder = len(lst) % m
    # Initialize the starting point for slicing the list
    start = 0
    parts = []
    for i in range(m):
        # Add an extra element to the current part if we have any remainder left
        end = start + part_size + (1 if remainder > 0 else 0)
        # Decrease the remainder count if we used one
        remainder -= 1 if remainder > 0 else 0
        # Append the current slice of the list to our parts list
        parts.append(lst[start:end])
        # The next slice will start where this one ended
        start = end
    return parts


def train_model(worker_num: int, k_models: [], epoch: int) -> ([], float):
    model_ids = []
    model_scores = []

    # 1. allocate models to works
    allocated_models = divide_list(k_models, worker_num)
    print(f"At Epoch {epoch}, worker host models are ", [len(ele) for ele in allocated_models])

    max_time = 0
    for worker_local_model in allocated_models:
        worker_exe_time = 0
        for model in worker_local_model:
            auc, _time = gtapi.get_valid_auc(model, epoch - 1)
            worker_exe_time += _time

            model_scores.append(auc)
            model_ids.append(model)
        # wait for the slower worker
        if worker_exe_time > max_time:
            max_time = worker_exe_time

    sorted_pairs = sorted(zip(model_scores, model_ids))
    sorted_model_scores, sorted_model_ids = zip(*sorted_pairs)
    return list(sorted_model_ids), max_time


def distributed_refinement(worker_num: int, k_models: []) -> float:
    total_time = 0
    total_rounds = int(math.log(len(k_models), 3))
    for round_index in range(1, total_rounds + 1):
        model_rank, time_usage = train_model(worker_num, k_models, 3 ** (round_index - 1))
        total_time += time_usage
        pick_num = int(len(k_models) / eta)
        k_models = model_rank[-pick_num:]
    return total_time


def draw_graph(worker_num, refinement_time, phase, x_label):
    mark_list = ["o", "*", "<", "^", "s", "d", "D", ">", "h"]
    line_shape_list = ['-.', '--', '-', ':']
    fig2, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(worker_num, refinement_time, line_shape_list[0] + mark_list[0],
            linewidth=4, markersize=20)
    ax.grid()
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(f"Total Time of the \n {phase} Phase (s)", fontsize=20)
    formatter = FuncFormatter(thousands_formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticks([1, 2, 4, 8])  # Specify which ticks you want
    ax.tick_params(axis='both', labelsize=20)
    print(f"./scale_{phase}.pdf")
    fig2.savefig(f"./scale_{phase.lower()}.pdf", bbox_inches='tight')


def seed_everything(seed=11):
    ''' [reference] https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335 '''
    random.seed(seed)


if __name__ == "__main__":

    seed_everything()

    # 2. compute and draw the refinement phase
    k_models = random_generate_models(40)
    worker_num = [1, 2, 4, 8]
    refinement_time = []
    for _worker_num in worker_num:
        query_latency = distributed_refinement(_worker_num, k_models)
        print(query_latency)
        refinement_time.append(query_latency)
    draw_graph(worker_num, refinement_time, "Refinement", "GPU Number")
