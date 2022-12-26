
from Config.Config                  import PathType
from pathlib                        import Path

from typing                         import Dict, Union, Optional, Any
from NeuralNetworks.NN_Base         import Model_Base
from Dataset.Dataset_Base           import Dataset_Base
from torch                          import optim

import torch



state_dict_type = Dict[str, Union[str, dict, int, float, Model_Base, Dataset_Base]]

'''
    Current saving dict format
'''


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


class SaveCheckpointUtils:
    """
        SaveCheckpointUtils class
    """

    def __init__(self, checkpoint_dict:Optional[state_dict_type]=None):
        self.checkpoint_dict = checkpoint_dict

    def __setitem__(self, key:str=None, value:Any=None):
        assert key is not None and value is not None, "key and value must be specified"
        self.checkpoint_dict[key] = value