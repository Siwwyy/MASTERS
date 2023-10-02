from Losses.Loss_Combined import Loss_Combined
from Losses.Loss_MAE import Loss_MAE
from Losses.Loss_MSE import Loss_MSE
from Losses.PerceptualLosses.PerceptualLoss_VGG import PerceptualLoss_VGG
from NeuralNetworks.NN_Base import Model_Base
from NeuralNetworks.UNet import Model_UNET
from NeuralNetworks.Model_Custom import Model_Custom
from NeuralNetworks.Model_NoCheckerboard import Model_NoCheckerboard
from Dataset.Dataset_Base import Dataset_Base
from Dataset.Dataset_UE import Dataset_UE, FullDataset_UE
from Config.Config import CurrentDevice, GetTrainingsPath, GetInferencePath

from torch.utils.data import DataLoader
from torch import optim
from typing import Any, Dict, Union, Optional
from pathlib import Path
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import inspect


@dataclass()
class ModelHyperparameters:
    in_channels: int = 3
    out_channels: int = 3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 15

    def __iter__(self):
        return iter(astuple(self))


# Default Configs Hyperparameters of model
HyperparametersDict = {
    "className": ModelHyperparameters,
    "args": {
        "in_channels": 3,
        "out_channels": 3,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "num_epochs": 600,
    },
}

# Core dict contains paths to folders, dtype used in model, device etc.
#model_stem = f"Model_NoCheckerboard/Epochs_{HyperparametersDict['args']['num_epochs']}_1_2_VGG600epochs_baseline_L1400Epochs"
#model_stem = f"Model_UNET/Epochs_{HyperparametersDict['args']['num_epochs']}_L1_SmallerChannels"
#model_stem = f"Model_UNET/Epochs_{HyperparametersDict['args']['num_epochs']}_L1"
model_stem = f"Model_NoCheckerboard/Epochs_{HyperparametersDict['args']['num_epochs']}_L1_VGG"
CoreDict = {
    "run_training": True,
    "load_model": False,
    "device": CurrentDevice,
    "dtype": torch.float32,
    "model_save_path": GetTrainingsPath(stem=model_stem),  # maybe use partial here
    "model_load_path": GetTrainingsPath(stem=model_stem),
    "model_inference_path": GetInferencePath(stem=model_stem),
    "cached_ds": True,
}

TrainDatasetDict = {
    "className": FullDataset_UE,
    "args": {
        "name": "FullDataset_UE",
        "ds_root_path": Path("F:/MASTERS/UE4/DATASET/"),
        "ue_projects_list": ["SubwaySequencer_4_26_2", "Rainforest_Scene_4_26_2"],
        "crop_coords": (900, 1028, 500, 628),
        # "crop_coords": (900, 964, 500, 564),
        "transforms": None,
        "cached": CoreDict["cached_ds"],
    },
}

ValidDatasetDict = {
    "className": Dataset_UE,
    "args": {
        "name": "Dataset_UE",
        "ds_root_path": Path(
            "F:/MASTERS/UE4/DATASET/InfiltratorDemo_4_26_2/DumpedBuffers"
        ),
        "csv_root_path": Path(
            "F:/MASTERS/UE4/DATASET/InfiltratorDemo_4_26_2/DumpedBuffers/info_Native.csv"
        ),
        "crop_coords": (900, 1028, 500, 628),
        # "crop_coords": (900, 964, 500, 564),
        "transforms": None,
        "cached": CoreDict["cached_ds"],
    },
}

TrainDataloaderDict = {
    "className": DataLoader,
    "args": {
        "dataset": None,
        "batch_size": HyperparametersDict["args"]["batch_size"],
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
    },
}

ValidDataloaderDict = {
    "className": DataLoader,
    "args": {
        "dataset": None,
        "batch_size": HyperparametersDict["args"]["batch_size"],
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
    },
}

ModelDict = {
    "className": Model_NoCheckerboard,
    "args": {
        "in_channels": HyperparametersDict["args"]["in_channels"],
        "out_channels": HyperparametersDict["args"]["out_channels"],
    },
}

#CriterionDict = {
#    "className": Loss_MSE,
#    "args": {},
#}

CriterionDict = {
    "className": Loss_Combined,
    "args": {
        "criterions": [Loss_MAE(), PerceptualLoss_VGG()],
        "criterionContribution": [1.0, 2.0],
        "device": CurrentDevice,
    },
}

OptimizerDict = {
    "className": optim.AdamW,
    "args": {"params": None, "lr": HyperparametersDict["args"]["learning_rate"]},
}

""" 
    Training, Dataset, Model etc. config, device, dtype utility
"""


class ConfigMapping(dict):
    """
    Config Mapping for config utility, stores a pre-defined
    values for training, inference, loss, model etc.

    e.g.,
    config[HyperparametersDict]   = HyperparametersDict['args']
    config[TrainDatasetDict]      = TrainDatasetDict['args']
    config['device']              = torch.device("cpu")
    """

    def __init__(self, mapping=None, *args: dict):
        if mapping is None:
            mapping = {}
        if args:
            for arg in args:
                mapping[str(arg["className"].__name__)] = arg["args"]

        super().__init__(mapping)

    def __getitem__(self, key: str = None) -> Any:
        assert key is not None
        return super().__getitem__(key)

    def __setitem__(self, key: [dict, str] = None, value: Union[dict, str, Any] = None):
        assert key is not None and value is not None, "key and value must be specified"
        # TODO Check if every argument is specified in value, otherwise set to None or raise error
        super().__setitem__(key, value)

    def __repr__(self) -> str:
        output_str = str()

        num_spaces = lambda len_str: 25 - len_str
        tabulation = "  "  # len == 2 -> is a amount of spaces, if value is dict, for better formatting e.g., tabulation (TAB, tabulation is usually a 4 spaces)
        for key, value in self.items():
            if type(value) is dict:
                output_str = output_str + f"{key} \n"
                spaces = " " * num_spaces(len("ClassName"))
                output_str = (
                    output_str
                    + f"{tabulation}ClassName: {spaces} {self[key]['className'].__name__} \n"
                )
                for subKey, subValue in self[key]["args"].items():
                    spaces = " " * num_spaces(len(subKey))
                    output_str = (
                        output_str + f"{tabulation}{subKey}: {spaces} {subValue} \n"
                    )
            else:
                spaces = " " * num_spaces(len(key) - len(tabulation))
                output_str = output_str + f"{key}: {spaces} {value} \n"

        return output_str


# Initialize config
# config = ConfigMapping(CoreDict)
# config["hyperparameters"] = HyperparametersDict
# config["trainDS"] = TrainDatasetDict
# config["validDS"] = ValidDatasetDict
# config["trainDL"] = TrainDataloaderDict
# config["validDL"] = ValidDataloaderDict
# config["model"] = ModelDict
# config["criterion"] = CriterionDict
# config["optimizer"] = OptimizerDict
# print(config)


def initObjectFromConfig(className: type, *args, **kwargs):
    """
    Initializes object from given config,

    e.g.,
    className == DataLoader
    kwargs ==
        'dataset':              None,
        'batch_size':           32,
        'shuffle':              True,
        'drop_last':            True,
        'pin_memory':           True

    Arg is optional, if args are specified, then function overwrites
    kwargs from beginning to len(args) - 1
    """

    if len(args) != 0:
        for idx, (kwargKey, kwargValue) in enumerate(kwargs.items()):
            if idx == len(args):
                break
            kwargs[kwargKey] = args[idx]

    return className(**kwargs)


def configJSONSerializer(config: ConfigMapping = None):
    assert config is not None, "config must be specified"
    import json

    with open(config["model_save_path"] / "training_config.json", "w") as convertFile:
        convertFile.write(str(config))
