from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
from src.tools.io_tools import read_json, write_json
import os

import logging

logging.getLogger('hpbandster').setLevel(logging.WARNING)

dataset_used = "frappe"
# dataset_used = "uci_diabetes"
# dataset_used = "criteo"

epoch_sampled = {"frappe": 13, "uci_diabetes": 0, "criteo": 9}
if dataset_used == "frappe":
    mlp_train_frappe = os.path.join(
        "/Users/kevin/project_python/VLDB_code/exp_data/",
        "tab_data/frappe/all_train_baseline_frappe.json")
    data_dict = read_json(mlp_train_frappe)
elif dataset_used == "criteo":
    mlp_train_criteo = os.path.join(
        "/Users/kevin/project_python/VLDB_code/exp_data/",
        "tab_data/criteo/all_train_baseline_criteo.json")
    data_dict = read_json(mlp_train_criteo)
elif dataset_used == "uci_diabetes":
    mlp_train_uci_diabetes = os.path.join(
        "/Users/kevin/project_python/VLDB_code/exp_data/",
        "tab_data/uci_diabetes/all_train_baseline_uci_160k_40epoch.json")
    data_dict = read_json(mlp_train_uci_diabetes)

performance_results = {}
for dataset, architectures in data_dict.items():
    for architecture, epochs in architectures.items():
        performance_results[architecture] = float(epochs[str(epoch_sampled[dataset])]["valid_auc"])


class MLPWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline_acc = []
        self.performance_results = performance_results

    def compute(self, config, budget, *args, **kwargs):
        architecture = '-'.join([str(config['layer_{}'.format(i)]) for i in range(4)])
        performance = self.performance_results.get(architecture, None)

        if len(self.baseline_acc) == 0:
            self.baseline_acc.append(performance)
        else:
            if performance > self.baseline_acc[-1]:
                self.baseline_acc.append(performance)
            else:
                self.baseline_acc.append(self.baseline_acc[-1])

        if performance is None:
            logging.warning('Performance result for architecture {} not found'.format(architecture))
            return {'loss': float('inf'), 'info': architecture}
        else:
            return {'loss': 1 - performance, 'info': architecture}


# Node options
# node_options = [8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 384, 512]

node_options = [8, 16, 32,
                48, 96, 112, 144, 176, 240,
                384]

# Configuration space
config_space = CS.ConfigurationSpace()
for i in range(4):
    config_space.add_hyperparameter(CSH.CategoricalHyperparameter('layer_{}'.format(i), choices=node_options))

result = {
    "sys_time_budget": [],
    "sys_acc": []
}

for i in range(50):
    print(i)
    # Start a nameserver
    host = 'localhost'
    port = 0
    ns = hpns.NameServer(run_id='example1', host=host, port=port)
    ns_host, ns_port = ns.start()

    # Start a worker
    w = MLPWorker(nameserver=ns_host, nameserver_port=ns_port, run_id='example1')
    w.run(background=True)

    # Start a BOHB optimizer
    bohb = BOHB(configspace=config_space, run_id='example1', nameserver=ns_host, nameserver_port=ns_port)
    res = bohb.run(n_iterations=180)

    # Shutdown
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    # Print the result
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    print('Best found configuration in iteration {0}:'.format(i), id2config[incumbent]['config'])

    result["sys_time_budget"].append(list(range(1, len(w.baseline_acc) + 1)))
    result["sys_acc"].append(w.baseline_acc)

checkpoint_file = f"./train_base_line_bohb_{dataset_used}_epoch_{epoch_sampled[dataset_used]}.json"
write_json(checkpoint_file, result)
