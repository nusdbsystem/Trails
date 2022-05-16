import random
import numpy as np
import os

os.environ.setdefault("base_dir", "../exp_data")
base_dir = os.path.join(os.getcwd(), "../exp_data")
from src.query_api.query_api_img import Gt201, Gt101
from src.tools.io_tools import write_json


def get_distribution_gt_201():
    acc = []
    data201 = Gt201()
    for i in range(data201.count_models()):
        test_acc, time_used = data201.get_c10_200epoch_test_info(arch_id=i)
        acc.append(test_acc)

    best_model = max(acc) * 0.01
    exp = np.array(acc)
    q_95 = np.quantile(exp, .95, axis=0).item() * 0.01
    q_85 = np.quantile(exp, .85, axis=0).item() * 0.01
    q_75 = np.quantile(exp, .75, axis=0).item() * 0.01
    q_65 = np.quantile(exp, .65, axis=0).item() * 0.01

    res = {"best": best_model, "q_95": q_95, "q_85": q_85, "q_75": q_75, "q_65": q_65}
    print(res)
    write_json(os.path.join(base_dir, "img_data/ground_truth/201_target"), res)


def get_distribution_gt_101():
    acc = []
    data101 = Gt101()
    for i in range(data101.count_models()):
        test_acc, time_used = data101.get_c10_test_info(arch_id=i)
        acc.append(test_acc)

    best_model = max(acc)
    exp = np.array(acc)
    q_95 = np.quantile(exp, .95, axis=0).item()
    q_85 = np.quantile(exp, .85, axis=0).item()
    q_75 = np.quantile(exp, .75, axis=0).item()
    q_65 = np.quantile(exp, .65, axis=0).item()

    res = {"best": best_model, "q_95": q_95, "q_85": q_85, "q_75": q_75, "q_65": q_65}
    print(res)
    write_json(os.path.join(base_dir, "img_data/ground_truth/101_target"), res)


def measure_if_overfitting():
    from matplotlib import pyplot as plt
    from src.third_pkg.sp201_lib import NASBench201API

    api_loc = os.path.join(base_dir, "data/NAS-Bench-201-v1_1-096897.pth")
    api = NASBench201API(api_loc)

    train_acc = []
    test_acc = []

    # this is measure if the 200 epoch is over fitting or not
    for archid in random.sample(range(15624), 400):
        for iepoch in range(200):
            info = api.get_more_info(archid, "cifar10-valid", iepoch=iepoch, hp="200", is_random=False)
            train_acc.append(info["train-accuracy"])
            test_acc.append(info["valid-accuracy"])

        plt.plot(train_acc, label="train-acc")
        plt.plot(test_acc, label="test-acc")
        plt.legend()
        plt.show()
        plt.clf()


# get the distribution of the ground truth, this is also the search target.
get_distribution_gt_101()
get_distribution_gt_201()
