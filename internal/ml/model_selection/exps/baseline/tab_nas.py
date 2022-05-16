
import numpy as np
import torch
from src.tools.io_tools import read_json, write_json
import os

DEFAULT_LAYER_CHOICES_20 = [8, 16, 24, 32,  # 8
                            48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256,  # 16
                            384, 512]
DEFAULT_LAYER_CHOICES_10 = [8, 16, 32,
                            48, 96, 112, 144, 176, 240,
                            384]


def get_parameter_number(hidden_layer_sizes, input_size=2):
    all_layer_sizes = [input_size] + list(hidden_layer_sizes) + [1]
    num_parameters = int(sum([all_layer_sizes[i] * all_layer_sizes[i + 1] + all_layer_sizes[i + 1] for i in
                              range(len(all_layer_sizes) - 1)]))
    return num_parameters


layer_1_choices = DEFAULT_LAYER_CHOICES_20
layer_2_choices = DEFAULT_LAYER_CHOICES_20
layer_3_choices = DEFAULT_LAYER_CHOICES_20
layer_4_choices = DEFAULT_LAYER_CHOICES_20

dataset_used = "frappe"
# dataset_used = "uci_diabetes"
# dataset_used = "criteo"

epoch_sampled = {"frappe": 13, "uci_diabetes": 0, "criteo": 9}
if dataset_used == "frappe":
    mlp_train_frappe = os.path.join(
        "../exp_data/",
        "tab_data/frappe/all_train_baseline_frappe.json")
    data_dict = read_json(mlp_train_frappe)
elif dataset_used == "criteo":
    mlp_train_criteo = os.path.join(
        "../exp_data/",
        "tab_data/criteo/all_train_baseline_criteo.json")
    data_dict = read_json(mlp_train_criteo)
elif dataset_used == "uci_diabetes":
    mlp_train_uci_diabetes = os.path.join(
        "../exp_data/",
        "tab_data/uci_diabetes/all_train_baseline_uci_160k_40epoch.json")
    data_dict = read_json(mlp_train_uci_diabetes)

rewards = {}
for dataset, architectures in data_dict.items():
    for architecture, epochs in architectures.items():
        arch_tuple = tuple([int(ele) for ele in architecture.split("-")])
        rewards[arch_tuple] = epochs[str(epoch_sampled[dataset])]["valid_auc"]

result_dir = "./internal/ml/model_selection/exp_result"
checkpoint_file = f"{result_dir}/tabNAS_benchmark_{dataset_used}_epoch_{epoch_sampled[dataset_used]}.json"

# hyperparameters
optimizer_name = 'adam'
num_samples_for_mc_start = 5
num_samples_for_mc_end = 5
rl_learning_rate = 0.1
if dataset_used == "frappe":
    max_iter = 19000
elif dataset_used == "criteo":
    max_iter = 5000
elif dataset_used == "uci_diabetes":
    max_iter = 9000


def run_sampling(i_rep):
    """
    Runs RL-based NAS with the rejection-based reward (TabNAS), starting from uniform distribution.

    Args:

    i_rep (int): the repetition index.

    Returns:

    layer_1_probs_all (dict): sampling probabilities of the first layer.
    layer_1_probs_all (dict): sampling probabilities of the second layer.
    """
    layer_1_probs_all = {}  # key: number of steps
    layer_2_probs_all = {}  # key: number of steps
    layer_3_probs_all = {}  # key: number of steps
    layer_4_probs_all = {}  # key: number of steps

    layer_1_logits = torch.zeros(len(layer_1_choices), requires_grad=True)
    layer_2_logits = torch.zeros(len(layer_2_choices), requires_grad=True)
    layer_3_logits = torch.zeros(len(layer_3_choices), requires_grad=True)
    layer_4_logits = torch.zeros(len(layer_4_choices), requires_grad=True)

    rl_reward_momentum = 0.9
    moving_average_baseline_numer = 0
    moving_average_baseline_denom = 0

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            [layer_1_logits, layer_2_logits, layer_3_logits, layer_4_logits],
            lr=rl_learning_rate, betas=(0.9, 0.999), eps=1e-8)

    rl_advantage_all = []
    prob_valid_all = []
    cur_best_performance = []

    for iter in range(max_iter):
        torch.manual_seed(1000 * i_rep + iter)

        num_samples_for_mc = (max_iter - iter) / max_iter * \
                    (num_samples_for_mc_start - num_samples_for_mc_end) + num_samples_for_mc_end
        num_samples_for_mc = int(np.ceil(num_samples_for_mc))

        layer_1_dist = torch.distributions.categorical.Categorical(logits=layer_1_logits)
        index_layer_1_choice = layer_1_dist.sample()
        layer_1_choice = layer_1_choices[index_layer_1_choice]

        layer_2_dist = torch.distributions.categorical.Categorical(logits=layer_2_logits)
        index_layer_2_choice = layer_2_dist.sample()
        layer_2_choice = layer_2_choices[index_layer_2_choice]

        layer_3_dist = torch.distributions.categorical.Categorical(logits=layer_3_logits)
        index_layer_3_choice = layer_3_dist.sample()
        layer_3_choice = layer_3_choices[index_layer_3_choice]

        layer_4_dist = torch.distributions.categorical.Categorical(logits=layer_4_logits)
        index_layer_4_choice = layer_4_dist.sample()
        layer_4_choice = layer_4_choices[index_layer_4_choice]

        layer_1_probs = layer_1_dist.probs.detach().numpy()
        layer_1_probs_all[iter] = layer_1_probs

        layer_2_probs = layer_2_dist.probs.detach().numpy()
        layer_2_probs_all[iter] = layer_2_probs

        layer_3_probs = layer_3_dist.probs.detach().numpy()
        layer_3_probs_all[iter] = layer_3_probs

        layer_4_probs = layer_4_dist.probs.detach().numpy()
        layer_4_probs_all[iter] = layer_4_probs

        if len(cur_best_performance) == 0:
            cur_best_performance.append(rewards[(layer_1_choice, layer_2_choice, layer_3_choice, layer_4_choice)])
        else:
            if rewards[(layer_1_choice, layer_2_choice, layer_3_choice, layer_4_choice)] > cur_best_performance[-1]:
                cur_best_performance.append(rewards[(layer_1_choice, layer_2_choice, layer_3_choice, layer_4_choice)])
            else:
                cur_best_performance.append(cur_best_performance[-1])
        # cur_time += time_usage[(layer_1_choice, layer_2_choice, layer_3_choice, layer_4_choice)]
        # cur_time_usage.append(cur_time)

        # compute single-step RL advantage
        moving_average_baseline_numer = \
            rl_reward_momentum * moving_average_baseline_numer + (1 - rl_reward_momentum) * \
            rewards[(layer_1_choice, layer_2_choice, layer_3_choice, layer_4_choice)]

        moving_average_baseline_denom = rl_reward_momentum * moving_average_baseline_denom + 1 - rl_reward_momentum
        moving_average_baseline = moving_average_baseline_numer / moving_average_baseline_denom
        rl_advantage = rewards[(layer_1_choice, layer_2_choice, layer_3_choice, layer_4_choice)] - moving_average_baseline
        rl_advantage_all.append(rl_advantage)

        layer_1_logits_tf = layer_1_dist.logits
        layer_2_logits_tf = layer_2_dist.logits

        # policy gradient via REINFORCE
        layer_1_logits_single_iter = layer_1_dist.logits  # torch.tensor
        layer_2_logits_single_iter = layer_2_dist.logits  # torch.tensor
        layer_3_logits_single_iter = layer_3_dist.logits  # torch tensor
        layer_4_logits_single_iter = layer_4_dist.logits  # torch tensor

        layer_1_probs_single_iter = torch.nn.functional.softmax(layer_1_logits_single_iter, dim=0)
        layer_2_probs_single_iter = torch.nn.functional.softmax(layer_2_logits_single_iter, dim=0)
        layer_3_probs_single_iter = torch.nn.functional.softmax(layer_3_logits_single_iter, dim=0)
        layer_4_probs_single_iter = torch.nn.functional.softmax(layer_4_logits_single_iter, dim=0)

        index_layer_1_samples_for_mc = layer_1_dist.sample(torch.Size([num_samples_for_mc]))
        index_layer_2_samples_for_mc = layer_2_dist.sample(torch.Size([num_samples_for_mc]))
        index_layer_3_samples_for_mc = layer_3_dist.sample(torch.Size([num_samples_for_mc]))
        index_layer_4_samples_for_mc = layer_4_dist.sample(torch.Size([num_samples_for_mc]))

        sampled_feasible_choices = [
            (index_layer_1_single, index_layer_2_single, index_layer_3_single, index_layer_4_single) for
            index_layer_1_single, index_layer_2_single, index_layer_3_single, index_layer_4_single in
            zip(index_layer_1_samples_for_mc, index_layer_2_samples_for_mc, index_layer_3_samples_for_mc, index_layer_4_samples_for_mc) \
            if (index_layer_1_single, index_layer_2_single, index_layer_3_single, index_layer_4_single) in rewards]

        if len(sampled_feasible_choices) == 0:
            continue
        estimated_prob_valid = torch.sum(torch.tensor(
            [layer_1_probs_single_iter[i] * layer_2_probs_single_iter[j]  * layer_3_probs_single_iter[k] * layer_3_probs_single_iter[n]
             for i, j, k, n in sampled_feasible_choices]))

        conditional_prob = layer_1_probs_single_iter[index_layer_1_choice]\
                           * layer_2_probs_single_iter[index_layer_2_choice] \
                           * layer_3_probs_single_iter[index_layer_3_choice] \
                           * layer_4_probs_single_iter[index_layer_4_choice] \
                           / estimated_prob_valid

        log_conditional_prob = torch.log(conditional_prob)
        negative_value_function = - float(rl_advantage) * log_conditional_prob

        negative_value_function.backward()
        optimizer.step()

    # get probs at final step
    layer_1_dist = torch.distributions.categorical.Categorical(logits=layer_1_logits)
    layer_2_dist = torch.distributions.categorical.Categorical(logits=layer_2_logits)
    layer_3_dist = torch.distributions.categorical.Categorical(logits=layer_3_logits)
    layer_4_dist = torch.distributions.categorical.Categorical(logits=layer_4_logits)

    layer_1_probs = layer_1_dist.probs.detach().numpy()
    layer_2_probs = layer_2_dist.probs.detach().numpy()
    layer_3_probs = layer_3_dist.probs.detach().numpy()
    layer_4_probs = layer_4_dist.probs.detach().numpy()

    layer_1_probs_all[max_iter] = layer_1_probs
    layer_2_probs_all[max_iter] = layer_2_probs
    layer_3_probs_all[max_iter] = layer_3_probs
    layer_4_probs_all[max_iter] = layer_4_probs

    return layer_1_probs_all, layer_2_probs_all, layer_3_probs_all, layer_4_probs_all, cur_best_performance


recorded_result = {
    "sys_time_budget": [],
    "sys_acc": []
}

n_reps = 50  # for easier demonstration; was 500 in paper
r = []
for i in range(n_reps):
    print(i)
    layer_1_probs_all, layer_2_probs_all, layer_3_probs_all, layer_4_probs_all, cur_best_performance = run_sampling(i)
    r.append([layer_1_probs_all, layer_2_probs_all, layer_3_probs_all, layer_4_probs_all])
    recorded_result["sys_time_budget"].append(list(range(1, len(cur_best_performance) + 1)))
    recorded_result["sys_acc"].append(cur_best_performance)


write_json(checkpoint_file, recorded_result)

