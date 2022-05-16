import ConfigSpace
import torch.nn as nn
from src.search_space.core.rl_policy import RLPolicyBase
import torch
NUM_VERTICES = 7
MAX_EDGES = 9


class RLPolicy101Topology(RLPolicyBase):
    def __init__(self, search_space, rl_learning_rate, max_nodes=4):
        super(RLPolicy101Topology, self).__init__()
        self.max_nodes = max_nodes
        self.search_space = search_space
        self.rl_learning_rate = rl_learning_rate

        self.cs = self.search_space.get_configuration_space()

        # 21 edges each cell
        self._edge_logits = nn.Parameter(1e-3 * torch.zeros(21, 2))
        # 3 ops in total
        self._op_logits = nn.Parameter(1e-3 * torch.zeros(5, 3))

        self.all_optimizer = torch.optim.Adam(self.parameters(), lr=rl_learning_rate)

    def generate_arch(self, config):
        while True:
            model_spec = self.search_space.config2arch_func(config)
            if self.search_space.api.is_valid(model_spec):
                return model_spec
            else:
                # resample a valid arch
                config, _, _ = self._sample_new_cfg()

    def select_action(self):
        config, sample, dists = self._sample_new_cfg()
        all_log_probs = [dists[i].log_prob(sample[i]) for i in range(len(sample))]
        return sum(all_log_probs), config

    def _sample_new_cfg(self):
        dists_edges = [torch.distributions.Categorical(logits=logit.unsqueeze(0)) for logit in self._edge_logits]
        dists_ops = [torch.distributions.Categorical(logits=logit.unsqueeze(0)) for logit in self._op_logits]

        dists = dists_edges + dists_ops
        sample = [di.sample() for di in dists]
        config = ConfigSpace.Configuration(self.cs, vector=sample)
        return config, sample, dists

    def update_policy(self, reward, baseline_values, log_prob):
        policy_loss = - log_prob * (reward - baseline_values)
        self.all_optimizer.zero_grad()
        policy_loss.backward()
        self.all_optimizer.step()
