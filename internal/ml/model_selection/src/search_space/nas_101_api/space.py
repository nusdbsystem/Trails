import copy
import itertools
import json
import random

import ConfigSpace
import numpy as np

from src.common.constant import Config
from src.search_space.core.model_params import ModelMicroCfg, ModelMacroCfg
from src.search_space.core.space import SpaceWrapper
from src.third_pkg.sp101_lib import nb101_api
from src.third_pkg.sp101_lib.model import NasBench101Network
from src.third_pkg.sp101_lib.nb101_api import ModelSpec
from src.search_space.nas_101_api.model_params import NB101MacroCfg
from src.search_space.nas_101_api.rl_policy import RLPolicy101Topology
from src.query_api.query_api_img import ImgScoreQueryApi, guess_train_one_epoch_time, guess_score_time
from src.query_api.interface import profile_NK_trade_off
from torch.utils.data import DataLoader
from typing import Generator

# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2  # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2  # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]  # Binary adjacency matrix


class NB101MicroCfg(ModelMicroCfg):

    @classmethod
    def builder(cls, encoding: str):
        data = json.loads(encoding)
        spec = ModelSpec(data["matrix"], data["operations"])
        return NB101MicroCfg(spec)

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec

    def __str__(self):
        return json.dumps({"matrix": self.spec.original_matrix.tolist(),
                           "operations": self.spec.original_ops})


class NasBench101Space(SpaceWrapper):
    api = None

    def __init__(self, api_loc: str, modelCfg: NB101MacroCfg, loapi: ImgScoreQueryApi):
        super().__init__(modelCfg, Config.NB101)
        self.api_loc = api_loc
        self.loapi = loapi

    def load(self):
        if NasBench101Space.api is None:
            NasBench101Space.api = nb101_api.NASBench(self.api_loc)

    def _is_valid(self, new_spec: ModelSpec):
        self.load()
        return self.api.is_valid(new_spec) and self._is_scored(new_spec)

    def _is_scored(self, new_spec: ModelSpec):
        arch_id = self._arch_to_id(new_spec)
        return self.loapi.is_arch_inside_data(arch_id)

    def _arch_to_id(self, arch_spec: object) -> str:
        assert isinstance(arch_spec, ModelSpec)
        self.load()
        if self.api.is_valid(arch_spec):
            arch_hash = arch_spec.hash_spec(ALLOWED_OPS)
            arch_id = list(self.api.hash_iterator()).index(arch_hash)
            return str(arch_id)
        else:
            return "-1"

    @classmethod
    def serialize_model_encoding(cls, arch_micro: ModelMicroCfg) -> str:
        assert isinstance(arch_micro, NB101MicroCfg)
        return str(arch_micro)

    @classmethod
    def deserialize_model_encoding(cls, model_encoding: str) -> ModelMicroCfg:
        return NB101MicroCfg.builder(model_encoding)

    @classmethod
    def new_arch_scratch(cls, arch_macro: ModelMacroCfg, arch_micro: ModelMicroCfg, bn: bool = True):
        assert isinstance(arch_micro, NB101MicroCfg)
        assert isinstance(arch_macro, NB101MacroCfg)
        model = NasBench101Network(arch_micro.spec,
                                   arch_macro.init_channels,
                                   arch_macro.num_stacks,
                                   arch_macro.num_modules_per_stack,
                                   arch_macro.num_labels,
                                   bn)

        return model

    def new_arch_scratch_with_default_setting(self, model_encoding: str, bn: bool):
        model_micro = NasBench101Space.deserialize_model_encoding(model_encoding)
        return NasBench101Space.new_arch_scratch(self.model_cfg, model_micro, bn)

    def profiling(self, dataset: str,
                  train_loader: DataLoader = None, val_loader: DataLoader = None,
                  args=None, is_simulate: bool = False) -> (float, float, int):
        score_time_per_model = guess_score_time(self.name, dataset)
        train_time_per_epoch = guess_train_one_epoch_time(self.name, dataset)
        N_K_ratio = profile_NK_trade_off(dataset)
        return score_time_per_model, train_time_per_epoch, N_K_ratio

    def profiling_score_time(
            self, dataset: str,
            train_loader: DataLoader = None, val_loader: DataLoader = None,
            args=None, is_simulate: bool = False):
        return guess_score_time(self.name, dataset)

    def profiling_train_time(self, dataset: str,
                             train_loader: DataLoader = None, val_loader: DataLoader = None,
                             args=None, is_simulate: bool = False):
        return guess_train_one_epoch_time(self.name, dataset)

    def micro_to_id(self, arch_struct: ModelMicroCfg) -> str:
        assert isinstance(arch_struct, NB101MicroCfg)
        arch_id = self._arch_to_id(arch_struct.spec)
        return str(arch_id)

    def new_architecture(self, arch_id: str):
        self.load()
        # id -> hash
        arch_hash = next(itertools.islice(self.api.hash_iterator(), int(arch_id), None))
        # arch_id = list(self.api.hash_iterator()).index(arch_hash)
        # hash -> spec
        matrix = self.api.fixed_statistics[arch_hash]['module_adjacency']
        operations = self.api.fixed_statistics[arch_hash]['module_operations']
        spec = ModelSpec(matrix, operations)
        # spec -> model

        assert isinstance(self.model_cfg, NB101MacroCfg)
        architecture = NasBench101Network(spec, self.model_cfg.init_channels,
                                          self.model_cfg.num_stacks,
                                          self.model_cfg.num_modules_per_stack,
                                          self.model_cfg.num_labels,
                                          self.model_cfg.bn)
        return architecture

    def new_architecture_with_micro_cfg(self, arch_micro: ModelMicroCfg):
        assert isinstance(arch_micro, NB101MicroCfg)
        spec = arch_micro.spec
        # generate network with adjacency and operation
        assert isinstance(self.model_cfg, NB101MacroCfg)
        architecture = NasBench101Network(spec, self.model_cfg.init_channels,
                                          self.model_cfg.num_stacks,
                                          self.model_cfg.num_modules_per_stack,
                                          self.model_cfg.num_labels,
                                          self.model_cfg.bn)
        return architecture

    def __len__(self):
        return len(self.api.hash_iterator())

    def get_arch_size(self, arch_micro: ModelMicroCfg) -> int:
        assert isinstance(arch_micro, NB101MicroCfg)
        return len(arch_micro.spec.matrix)

    def sample_all_models(self) -> Generator[str, ModelMicroCfg, None]:
        self.load()
        total_num_arch = len(self.api.hash_iterator())
        arch_id_list = random.sample(range(total_num_arch), total_num_arch)

        for arch_id in arch_id_list:
            arch_hash = next(itertools.islice(self.api.hash_iterator(), int(arch_id), None))
            matrix = self.api.fixed_statistics[arch_hash]['module_adjacency']
            operations = self.api.fixed_statistics[arch_hash]['module_operations']
            spec = ModelSpec(matrix, operations)
            model_micro = NB101MicroCfg(spec)
            yield str(arch_id), model_micro

    def random_architecture_id(self) -> (str, ModelMicroCfg):
        """Returns a random valid spec."""
        while True:
            matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = ModelSpec(matrix=matrix, ops=ops)
            if self._is_valid(spec):
                arch_id = self._arch_to_id(spec)
                return str(arch_id), NB101MicroCfg(spec)

    '''Below is for EA'''

    def mutate_architecture(self, parent_arch: ModelMicroCfg) -> (str, ModelMicroCfg):
        mutation_rate = 1.0
        assert isinstance(parent_arch, NB101MicroCfg)
        self.load()
        """Computes a valid mutated spec from the old_spec."""
        while True:
            new_matrix = copy.deepcopy(parent_arch.spec.original_matrix)
            new_ops = copy.deepcopy(parent_arch.spec.original_ops)

            # In expectation, V edges flipped (note that most end up being pruned).
            edge_mutation_prob = mutation_rate / NUM_VERTICES
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            # In expectation, one op is resampled.
            op_mutation_prob = mutation_rate / OP_SPOTS
            for ind in range(1, NUM_VERTICES - 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in self.api.config['available_ops'] if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            new_spec = ModelSpec(new_matrix, new_ops)
            if self._is_valid(new_spec) and self._is_scored(new_spec):
                arch_id = self._arch_to_id(new_spec)
                return arch_id, NB101MicroCfg(new_spec)

    '''Below is for RL and BOHB'''

    def get_reinforcement_learning_policy(self, rl_learning_rate):
        return RLPolicy101Topology(self, rl_learning_rate, NUM_VERTICES)

    def get_configuration_space(self):
        cs = ConfigSpace.ConfigurationSpace()
        ops_choices = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices))
        cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices))
        for i in range(NUM_VERTICES * (NUM_VERTICES - 1) // 2):
            cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1]))
        return cs

    def config2arch_func(self, config):
        matrix = np.zeros([NUM_VERTICES, NUM_VERTICES], dtype=np.int8)
        idx = np.triu_indices(matrix.shape[0], k=1)
        for i in range(NUM_VERTICES * (NUM_VERTICES - 1) // 2):
            row = idx[0][i]
            col = idx[1][i]
            matrix[row, col] = config["edge_%d" % i]
        labeling = [config["op_node_%d" % i] for i in range(5)]
        labeling = ['input'] + list(labeling) + ['output']
        model_spec = ModelSpec(matrix, labeling)
        return model_spec
