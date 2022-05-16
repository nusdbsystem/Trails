import os
from src.common.constant import Config
from src.search_space.core.space import SpaceWrapper
from src.query_api.query_api_img import ImgScoreQueryApi


def init_search_space(args) -> SpaceWrapper:
    """
    :param args:
    :param loapi: Local score API, records all scored arch, 101 use it to detect which arch is scored.
    :return:
    """

    if args.search_space == Config.NB101:
        from .nas_101_api.model_params import NB101MacroCfg
        from .nas_101_api.space import NasBench101Space
        model_cfg = NB101MacroCfg(
            args.init_channels,
            args.num_stacks,
            args.num_modules_per_stack,
            args.num_labels)

        base_dir_folder = args.base_dir
        local_api = ImgScoreQueryApi(Config.NB101, args.dataset)
        return NasBench101Space(
            api_loc=os.path.join(base_dir_folder, "data", args.api_loc),
            modelCfg=model_cfg,
            loapi=local_api)

    elif args.search_space == Config.NB201:
        from .nas_201_api.model_params import NB201MacroCfg
        from .nas_201_api.space import NasBench201Space
        model_cfg = NB201MacroCfg(
            args.init_channels,
            args.init_b_type,
            args.init_w_type,
            args.arch_size,
            args.num_labels)

        base_dir_folder = args.base_dir
        return NasBench201Space(os.path.join(base_dir_folder, "data", args.api_loc), model_cfg)

    elif args.search_space == Config.MLPSP:
        from .mlp_api.space import MlpSpace
        from .mlp_api.model_params import MlpMacroCfg
        from .mlp_api.space import DEFAULT_LAYER_CHOICES_20, DEFAULT_LAYER_CHOICES_10
        if args.hidden_choice_len == 10:
            model_cfg = MlpMacroCfg(
                args.nfield,
                args.nfeat,
                args.nemb,
                args.num_layers,
                args.num_labels,
                DEFAULT_LAYER_CHOICES_10)
        else:
            model_cfg = MlpMacroCfg(
                args.nfield,
                args.nfeat,
                args.nemb,
                args.num_layers,
                args.num_labels,
                DEFAULT_LAYER_CHOICES_20)

        return MlpSpace(model_cfg)
    else:
        raise Exception

    # elif args.nasspace == 'nds_resnet':
    #     return NDS('ResNet')
    # elif args.nasspace == 'nds_amoeba':
    #     return NDS('Amoeba')
    # elif args.nasspace == 'nds_amoeba_in':
    #     return NDS('Amoeba_in')
    # elif args.nasspace == 'nds_darts_in':
    #     return NDS('DARTS_in')
    # elif args.nasspace == 'nds_darts':
    #     return NDS('DARTS')
    # elif args.nasspace == 'nds_darts_fix-w-d':
    #     return NDS('DARTS_fix-w-d')
    # elif args.nasspace == 'nds_darts_lr-wd':
    #     return NDS('DARTS_lr-wd')
    # elif args.nasspace == 'nds_enas':
    #     return NDS('ENAS')
    # elif args.nasspace == 'nds_enas_in':
    #     return NDS('ENAS_in')
    # elif args.nasspace == 'nds_enas_fix-w-d':
    #     return NDS('ENAS_fix-w-d')
    # elif args.nasspace == 'nds_pnas':
    #     return NDS('PNAS')
    # elif args.nasspace == 'nds_pnas_fix-w-d':
    #     return NDS('PNAS_fix-w-d')
    # elif args.nasspace == 'nds_pnas_in':
    #     return NDS('PNAS_in')
    # elif args.nasspace == 'nds_nasnet':
    #     return NDS('NASNet')
    # elif args.nasspace == 'nds_nasnet_in':
    #     return NDS('NASNet_in')
    # elif args.nasspace == 'nds_resnext-a':
    #     return NDS('ResNeXt-A')
    # elif args.nasspace == 'nds_resnext-a_in':
    #     return NDS('ResNeXt-A_in')
    # elif args.nasspace == 'nds_resnext-b':
    #     return NDS('ResNeXt-B')
    # elif args.nasspace == 'nds_resnext-b_in':
    #     return NDS('ResNeXt-B_in')
    # elif args.nasspace == 'nds_vanilla':
    #     return NDS('Vanilla')
    # elif args.nasspace == 'nds_vanilla_lr-wd':
    #     return NDS('Vanilla_lr-wd')
    # elif args.nasspace == 'nds_vanilla_lr-wd_in':
    #     return NDS('Vanilla_lr-wd_in')
