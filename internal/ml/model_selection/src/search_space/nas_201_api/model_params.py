from src.search_space.core.model_params import ModelMacroCfg


class NB201MacroCfg(ModelMacroCfg):

    def __init__(self, init_channels, init_b_type, init_w_type, max_nodes, num_labels):
        super().__init__(num_labels)
        self.init_channels = init_channels
        self.init_b_type = init_b_type
        self.init_w_type = init_w_type
        self.max_nodes = max_nodes
