from .MSNet import MSNet
from .TurbSRNet import FastTurbNet


def build_model(model_name: str, param_dict: dict):
    models = {
        'MSNet': MSNet,
        'FastNet': FastTurbNet
    }

    if model_name not in models.keys():
        raise ValueError(f'No such model: {model_name}!')

    return models[model_name](**param_dict)
