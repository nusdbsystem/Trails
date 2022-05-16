from src.search_space.core.rl_policy import RLPolicyBase
from copy import deepcopy
import torch
import torch.nn as nn
from src.third_pkg.models import CellStructure
from torch.distributions import Categorical


class RLPolicy201Topology(RLPolicyBase):
    def __init__(self, search_space, rl_learning_rate, max_nodes=4):
        super(RLPolicy201Topology, self).__init__()
        self.max_nodes = max_nodes
        self.search_space = deepcopy(search_space)
        self.edge2index = {}
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                self.edge2index[node_str] = len(self.edge2index)
        self.arch_parameters = nn.Parameter(
            1e-3 * torch.randn(len(self.edge2index), len(search_space))
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=rl_learning_rate)

    def generate_arch(self, actions):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = self.search_space[actions[self.edge2index[node_str]]]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.search_space[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def forward(self):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        return alphas

    def select_action(self):
        probs = self.forward()
        m = Categorical(probs)
        action = m.sample()
        # policy.saved_log_probs.append(m.log_prob(action))
        log_prob = m.log_prob(action)
        return log_prob, action.cpu().tolist()

    def update_policy(self, reward, baseline_values, log_prob):
        policy_loss = (-log_prob * (reward - baseline_values)).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
