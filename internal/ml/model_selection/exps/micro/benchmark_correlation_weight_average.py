import random
import os
from typing import List


# Initialize function to calculate correlation
def calculate_correlation(dataset, search_space, epoch_train, srcc_top_k: List = [1], is_visual=False):
    print("\n================================================================")
    print(f" {dataset} + {search_space}")
    print("================================================================")
    # Initialize query objects
    acc_query = SimulateTrain(space_name=search_space)
    score_query = SimulateScore(space_name=search_space, dataset_name=dataset)

    # Get list of all model IDs that have been trained and scored
    trained_models = acc_query.query_all_model_ids(dataset)
    scored_models = score_query.query_all_model_ids(dataset)

    # Find the intersection between trained_models and scored_models
    trained_scored_models = list(set(trained_models) & set(scored_models))

    # each alg have one corresponding ground truth list
    model_train_res_lst = {"nas_wot_add_syn_flow": []}
    all_alg_score_dic = {"nas_wot_add_syn_flow": []}

    data_points = []
    # Populate algorithm scores and training results
    for model_id in trained_scored_models:
        score_value = score_query.query_all_tfmem_score(model_id)
        acc, _ = acc_query.get_ground_truth(arch_id=model_id, dataset=dataset, epoch_num=epoch_train)

        f1 = float(score_value["nas_wot"])
        f2 = float(score_value["synflow"])
        y_pred = (0.00000000e+00 +
                  5.77392575e+06 * f1 +
                  -6.05999825e+04 * f2 +
                  6.61946937e+05 * f1 ** 2 +
                  5.33556789e+05 * f2 ** 2 +
                  6.12780778e-03 * f1 * f2)

        data_points.append(
            [acc, f1, f2]
        )
    visualization(data_points)


def visualization(data_points):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Coefficients a and b
    f1 = []
    f2 = []
    y = []
    for ele in data_points:
        f1.append(ele[1])
        f2.append(ele[2])
        y.append(ele[0])
    f1 = np.array(f1)
    f2 = np.array(f2)
    y = np.array(y)

    # Creating a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot - plots a dot for each set of (f1, f2, y)
    ax.scatter(f1, f2, y)

    # Setting the labels for each axis
    ax.set_xlabel('Feature 1 (f1)')
    ax.set_ylabel('Feature 2 (f2)')
    ax.set_zlabel('Label (y)')

    # Display the plot
    plt.show()


# Call the main function
if __name__ == "__main__":
    os.environ.setdefault("base_dir", "../exp_data")
    from src.query_api.interface import SimulateTrain, SimulateScore
    from src.tools.correlation import CorCoefficient
    from src.common.constant import CommonVars, Config

    # Frappe configuration, here also measure SRCC of top 0.2% -> 0.8%
    calculate_correlation(Config.Frappe, Config.MLPSP, 13, srcc_top_k=[1], is_visual=False)

    # UCI configuration
    calculate_correlation(Config.UCIDataset, Config.MLPSP, 0, is_visual=False)

    # Criteo configuration
    calculate_correlation(Config.Criteo, Config.MLPSP, 9, is_visual=False)

    # NB101 + C10
    calculate_correlation(Config.c10, Config.NB101, None)

    # NB201 + C10
    calculate_correlation(Config.c10, Config.NB201, None)

    # NB201 + C100
    calculate_correlation(Config.c100, Config.NB201, None)

    # NB201 + imageNet
    calculate_correlation(Config.imgNet, Config.NB201, None)
