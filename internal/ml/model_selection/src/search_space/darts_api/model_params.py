
from src.search_space.core.model_params import ModelMacroCfg


class DartsCfg(ModelMacroCfg):

    def __init__(self, dataset, bn, init_channels, layers, auxiliary, max_sample):
        '''
        :param dataset:
        :param bn:
        :param init_channels:
        :param layers:
        :param auxiliary:
        :param max_sample: max number of architectures used
        '''
        super(DartsCfg, self).__init__(init_channels)
        self.dataset = dataset

        if dataset == 'cifar10':
            self.NUM_CLASSES = 10
        elif dataset == 'cifar100':
            self.NUM_CLASSES = 100
        elif dataset == 'imagenet':
            self.NUM_CLASSES = 1000
        else:
            raise ValueError('Donot support dataset %s' % dataset)

        self.layers = layers
        self.auxiliary = auxiliary

        self.max_sample = max_sample
