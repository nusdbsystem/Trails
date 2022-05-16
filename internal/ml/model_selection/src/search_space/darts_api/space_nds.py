import json
import pickle
from src.search_space.darts_api.model_params import DartsCfg
import codecs
from src.search_space.darts_api.space_darts import DartsSpace


class NDSSpace(DartsSpace):
    """
    NDS have 5k images sampled from darts, each one has a full training.valid info
    """
    def __init__(self, api_loc: str, modelCfg: DartsCfg):
        super().__init__(modelCfg)
        # api_loc should be
        # DARTS.json:  for cifar
        # DARTS_fix-w-d.json: fixed weight-decay, cifar
        # DARTS_in.json: for imagenet only
        self.nds_api = json.load(open(f'nds_data/{api_loc}.json', 'r'))

    def random_architecture_id(self, max_nodes: int) -> (str, object):
        # 28 edges and 7 operations
        size = [14 * 2, 7]

        for uid in self.nds_api:
            netinfo = self.nds_api[uid]
            config = netinfo['net']
            gen = config['genotype']
            genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'],
                                reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])

            # update config accordingly
            self.model_cfg.init_channels = config['width']
            # self.model_cfg.NUM_CLASSES = 1
            self.model_cfg.layers = config['depth']
            self.model_cfg.auxiliary = config['aux']

            obj_base64string = codecs.encode(pickle.dumps(genotype, protocol=pickle.HIGHEST_PROTOCOL),
                                             "base64").decode(
                'latin1')

            yield obj_base64string, genotype

