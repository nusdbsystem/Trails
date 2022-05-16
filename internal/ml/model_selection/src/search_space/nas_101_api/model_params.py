from src.search_space.core.model_params import ModelMacroCfg


class NB101MacroCfg(ModelMacroCfg):

    def __init__(self, init_channels, num_stacks, num_modules_per_stack, num_labels):
        super().__init__(num_labels)
        self.init_channels = init_channels
        self.num_stacks = num_stacks
        self.num_modules_per_stack = num_modules_per_stack
