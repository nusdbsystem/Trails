from src.common.constant import Config


def get_base_annotations(dataset_name):
    """
    This is from http://proceedings.mlr.press/v139/xu21m/xu21m.pdf
    :param dataset_name:
    :return:
    """
    if dataset_name == Config.c10:
        annotations = [
            # ["REA (Training-based)", 93.92, 12000],
            # ["RS (Training-based)", 93.70, 12000],
            # ["REINFORCE (Training-based)", 93.85, 12000],
            # ["BOHB (Training-based)", 93.61, 12000],

            ["ENAS (Weight sharing)", 54.30, 13314.51],
            ["DARTS-V1 (Weight sharing)", 54.30, 16281],
            ["DARTS-V2", 54.30, 43277],

            # ["NASWOT (Training-Free)", 92.96, 306],
            ["TE-NAS (Training-Free)", 93.90, 2200],
            ["KNAS (Training-Free)", 93.38, 4400],
        ]
    elif dataset_name == Config.c100:
        annotations = [
            # ["REA (Training-based)", 93.92, 12000],
            # ["RS (Training-based)", 93.70, 12000],
            # ["REINFORCE (Training-based)", 93.85, 12000],
            # ["BOHB (Training-based)", 93.61, 12000],

            ["ENAS (Weight sharing)", 15.61, 13314.51],
            ["DARTS-V1 (Weight sharing)", 15.61, 16281],
            ["DARTS-V2", 15.61, 43277],

            # ["NASWOT (Training-Free)", 69.98, 306],
            ["TE-NAS (Training-Free)", 71.24, 4600],
            ["KNAS (Training-Free)", 70.78, 9200],
        ]
    elif dataset_name == Config.imgNet:
        annotations = [
            # ["REA (Training-based)", 93.92, 12000],
            # ["RS (Training-based)", 93.70, 12000],
            # ["REINFORCE (Training-based)", 93.85, 12000],
            # ["BOHB (Training-based)", 93.61, 12000],

            ["ENAS (Weight sharing)", 16.32, 13314.51],
            ["DARTS-V1 (Weight sharing)", 16.32, 16281],
            ["DARTS-V2", 16.32, 43277],

            # ["NASWOT (Training-Free)", 44.44, 306],
            ["TE-NAS (Training-Free)", 42.38, 10000],
            ["KNAS (Training-Free)", 44.63, 20000],
        ]
    else:
        annotations = []
    return annotations
