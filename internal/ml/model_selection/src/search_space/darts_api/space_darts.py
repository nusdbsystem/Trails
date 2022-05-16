import pickle
import numpy as np
from src.common.constant import Config
from src.search_space.darts_api.model_params import DartsCfg
import codecs
from src.search_space.core.space import SpaceWrapper


class DartsSpace(SpaceWrapper):

    def __init__(self, modelCfg: DartsCfg):
        super().__init__(modelCfg, Config.DARTS)

    def new_architecture(self, arch_id: str):
        return self.new_architecture_hash(arch_id)

    def new_architecture_hash(self, arch_hash: str):
        arch_geno = pickle.loads(codecs.decode(arch_hash.encode('latin1'), "base64"))
        if self.modelCfg.dataset == 'cifar10':
            model = NetworkCIFAR(self.modelCfg.init_channels, self.modelCfg.NUM_CLASSES,
                                 self.modelCfg.layers, self.modelCfg.auxiliary, arch_geno)
        elif self.modelCfg.dataset == 'cifar100':
            model = NetworkCIFAR(self.modelCfg.init_channels, self.modelCfg.NUM_CLASSES,
                                 self.modelCfg.layers, self.modelCfg.auxiliary, arch_geno)
        elif self.modelCfg.dataset == 'imagenet':
            model = NetworkImageNet(self.modelCfg.init_channels, self.modelCfg.NUM_CLASSES,
                                    self.modelCfg.layers, self.modelCfg.auxiliary, arch_geno)
        else:
            raise

        return model

    def __len__(self):
        return len(self.modelCfg.max_sample)

    def get_arch_size(self, architecture) -> int:
        return len(architecture.spec.matrix)

    def random_architecture_id(self, max_nodes: int) -> (str, object):
        # 28 edges and 7 operations
        size = [14 * 2, 7]
        # random weight for each ops in each edge
        random_weight = np.random.random_sample(size)
        one_new_geno = genotype(random_weight.reshape(2, -1, size[-1]))
        obj_base64string = codecs.encode(pickle.dumps(one_new_geno, protocol=pickle.HIGHEST_PROTOCOL), "base64").decode(
            'latin1')
        return obj_base64string, one_new_geno

    def mutate_architecture(self, parent_arch: object) -> object:
        """
         This will mutate one op from the parent op indices, and then
         update the naslib object and op_indices
         """

        # parent_arch is genotype,
        mutation_rate = 1
        parent_compact = convert_genotype_to_compact(parent_arch)
        parent_compact = make_compact_mutable(parent_compact)
        compact = parent_compact

        for _ in range(int(mutation_rate)):
            cell = np.random.choice(2)
            pair = np.random.choice(8)
            num = np.random.choice(2)
            if num == 1:
                compact[cell][pair][num] = np.random.choice(NUM_OPS)
            else:
                inputs = pair // 2 + 2
                choice = np.random.choice(inputs)
                if pair % 2 == 0 and compact[cell][pair + 1][num] != choice:
                    compact[cell][pair][num] = choice
                elif pair % 2 != 0 and compact[cell][pair - 1][num] != choice:
                    compact[cell][pair][num] = choice

        # return a new genotype
        return convert_compact_to_genotype(compact)

    def arch_to_id(self, arch_struct: object) -> str:
        obj_base64string = codecs.encode(pickle.dumps(arch_struct, protocol=pickle.HIGHEST_PROTOCOL), "base64").decode(
            'latin1')
        return obj_base64string
