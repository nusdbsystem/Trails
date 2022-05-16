import itertools
import os
import random

from src.third_pkg.sp101_lib import nb101_api
from src.tools.io_tools import write_json, write_pickle


def _get_spec(api, arch_hash: str):
    matrix = api.fixed_statistics[arch_hash]['module_adjacency']
    operations = api.fixed_statistics[arch_hash]['module_operations']
    spec = nb101_api.ModelSpec(matrix, operations)
    return spec


def query_api(api, arch_hash, epoch_num):
    res = api.query(_get_spec(api, arch_hash), epochs=epoch_num)
    static = {
        "architecture_id": arch_hash,
        "trainable_parameters": res["trainable_parameters"],
        "time_usage": res["training_time"],
        "train_accuracy": res["train_accuracy"],
        "validation_accuracy": res["validation_accuracy"],
        "test_accuracy": res["test_accuracy"],
    }
    return static


def save_best_score(api_used):
    total_num_arch = len(api_used.hash_iterator())
    arch_id_list = random.sample(range(total_num_arch), total_num_arch)

    parsed_result = {}
    for arch_id in arch_id_list:
        parsed_result[arch_id] = {}
        parsed_result[arch_id]["cifar10"] = {}
        arch_hash = next(itertools.islice(api_used.hash_iterator(), arch_id, None))
        for epoch in [4, 12, 36, 108]:
            query_info = query_api(api_used, arch_hash, epoch)
            parsed_result[arch_id]["cifar10"][epoch] = {
                "architecture_id": query_info["architecture_id"],
                "trainable_parameters": query_info["trainable_parameters"],
                "train_accuracy": query_info["train_accuracy"],
                "validation_accuracy": query_info["validation_accuracy"],
                'test-accuracy': query_info["test_accuracy"],
                'time_usage': query_info["time_usage"]
            }

    import pickle
    write_pickle("101_allEpoch_info_pickle", parsed_result)
    print("writing pickle done!")
    write_json("101_allEpoch_info_json", parsed_result)


def save_id_to_hash_dict(api_used):
    total_num_arch = len(api_used.hash_iterator())
    arch_id_list = random.sample(range(total_num_arch), total_num_arch)

    id_to_hash = {}
    for arch_id in arch_id_list:
        arch_hash = next(itertools.islice(api_used.hash_iterator(), arch_id, None))
        id_to_hash[arch_id] = arch_hash
    write_json("nb101_id_to_hash.json", id_to_hash)


if __name__ == "__main__":
    base_dir = os.getcwd()
    api_loc = os.path.join(base_dir, "data/nasbench_full.tfrecord")
    apifull = nb101_api.NASBench(api_loc)
    save_best_score(apifull)

    # api_loc = os.path.join(base_dir, "data/nasbench_only108.pkl")
    # api108 = nb101_api.NASBench(api_loc)
    # save_id_to_hash_dict(api108)
