import os
from src.tools.io_tools import write_json
from nats_bench import create


# obtain the metric for the `index`-th architecture
# `dataset` indicates the dataset:
#   'cifar10-valid'  : using the proposed train set of CIFAR-10 as the training set
#   'cifar10'        : using the proposed train+valid set of CIFAR-10 as the training set
#   'cifar100'       : using the proposed train set of CIFAR-100 as the training set
#   'ImageNet16-120' : using the proposed train set of ImageNet-16-120 as the training set
# `iepoch` indicates the index of training epochs from 0 to 11/199.
#   When iepoch=None, it will return the metric for the last training epoch
#   When iepoch=11, it will return the metric for the 11-th training epoch (starting from 0)
# `use_12epochs_result` indicates different hyper-parameters for training
#   When use_12epochs_result=True, it trains the network with 12 epochs and the LR decayed from 0.1 to 0 within 12 epochs
#   When use_12epochs_result=False, it trains the network with 200 epochs and the LR decayed from 0.1 to 0 within 200 epochs
# `is_random`
#   When is_random=True, the performance of a random architecture will be returned
#   When is_random=False, the performanceo of all trials will be averaged.


def simulate_train_eval(index, dataset, iepoch, hp, is_random=False):
    info = api.get_more_info(index, dataset, iepoch=iepoch, hp=hp, is_random=is_random)
    if dataset == "cifar10":
        test_acc = info["test-accuracy"]
        time_usage = info["train-all-time"] + info["test-per-time"]
        return test_acc, time_usage
    if dataset == "cifar10-valid":
        test_acc = info["valid-accuracy"]
        time_usage = info["train-all-time"] + info["valid-per-time"]
        return test_acc, time_usage
    if dataset == 'cifar100':
        test_acc = info["valtest-accuracy"]
        time_usage = info["train-all-time"] + info['valtest-per-time']
        return test_acc, time_usage
    if dataset == 'ImageNet16-120':
        test_acc = info["valtest-accuracy"]
        time_usage = info["train-all-time"] + info['valtest-per-time']
        return test_acc, time_usage


def save_acc_time():
    parsed_result = {}
    for arch_id in range(0, 15625):
        print(arch_id)
        parsed_result[arch_id] = {}
        for epoch_num in [12, 200]:
            parsed_result[arch_id][epoch_num] = {}
            for dataset in ['cifar10', 'cifar10-valid', 'cifar100', 'ImageNet16-120']:
                parsed_result[arch_id][epoch_num][dataset] = {}
                for each_epoch in range(epoch_num):
                    test_acc, time_usage = simulate_train_eval(arch_id, dataset, iepoch=each_epoch, hp=str(epoch_num))
                    parsed_result[arch_id][epoch_num][dataset][each_epoch] = {
                        "test_accuracy": test_acc,
                        "time_usage": time_usage
                    }
    write_json(pre_file, parsed_result)


if __name__ == "__main__":
    base_dir = os.getcwd()
    api_loc = os.path.join(base_dir, "data/NAS-Bench-201-v1_1-096897.pth")
    pre_file = os.path.join(base_dir, "result_base/ground_truth/201_allEpoch_info")

    api = create(None, "tss", fast_mode=True, verbose=False)

    simulate_train_eval(9099, "cifar10", iepoch=None, hp=str(200))

    save_acc_time()
