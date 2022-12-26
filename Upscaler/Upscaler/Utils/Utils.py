
from Config.Config import PathType
from pathlib import Path
from typing import Dict, Union
from NeuralNetworks.NN_Base import NN_Base

import torch



state_dict_type = Dict[str, Union[str, dict, int, float]]

def load_model(load_path:PathType=None) -> state_dict_type:
    assert load_path is not None, "Path can't be None! Please specify absolute path to save model"

    if type(load_path) is str:
        load_path = Path(load_path)

    return torch.load(load_path)



def save_model(save_path:PathType=None, training_state_dict:state_dict_type=None):
    assert save_path is not None, "Path can't be None! Please specify absolute path to save model"
    assert training_state_dict is not None, "Training state dict must be specified"

    if type(save_path) is str:
        save_path = Path(save_path)
    if not (save_path.parent).exists():
        (save_path.parent).mkdir(exist_ok=True)

    torch.save(training_state_dict, save_path)
