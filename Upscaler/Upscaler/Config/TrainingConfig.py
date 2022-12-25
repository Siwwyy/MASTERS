
from NeuralNetworks.NN_Base             import NN_Base
from NeuralNetworks.UNet                import Model_UNET
from NeuralNetworks.Model_Custom        import Model_Custom
from Dataset.Dataset_Base               import Dataset_Base
from Dataset.Dataset_UE                 import Dataset_UE
from Config.Config                      import CurrentDevice

from dataclasses                        import dataclass,astuple
from torch.utils.data                   import DataLoader
from torch                              import optim
from typing                             import Any, Dict
from pathlib                            import Path

import torch
import torch.nn as nn



TrainingDictType = Dict[str, Any]

@dataclass()
class ModelHyperparameters:
    in_channels :int = 3
    out_channels :int = 3
    learning_rate :float = 0.001
    batch_size :int = 1
    num_epochs :int = 15

    def __iter__(self):
        return iter(astuple(self))


def GetDefaultTrainingDict() -> TrainingDictType:
    """
    Returns: Default Training Configuration Dictonary
    """
    """Default Training Configuration Dictonary

    Returns
    -------
        Default Training Configuration Dictonary."""

    dtype = torch.float32
    hyperparams = ModelHyperparameters()
    # Create Dataset
    train_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
                          csv_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"),
                          #crop width x height == 128x128 (for now)
                          crop_coords=(900, 1028, 500, 628))
    # Create dataloader
    train_loader = DataLoader(dataset=train_ds, batch_size=hyperparams.batch_size, drop_last=True,                                    pin_memory=True)

    # Initialize network
    #model = Model_UNET(in_channels=ModelHyperparameters.in_channels, 
    #                   out_channels=ModelHyperparameters.out_channels).to(device=CurrentDevice, dtype=dtype)

    model = Model_Custom(in_channels=ModelHyperparameters.in_channels, 
                       out_channels=ModelHyperparameters.out_channels).to(device=CurrentDevice, dtype=dtype)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

    return {
        'hyperparams':          hyperparams,
        'train_ds':             train_ds,
        'train_dataloader':     train_loader,
        'model':                model,
        'criterion':            criterion,
        'optimizer':            optimizer,
        'device':               CurrentDevice,
        'dtype':                dtype,
    }
    

DefaultTrainingDict :TrainingDictType = GetDefaultTrainingDict()



""" 
    Training config, device, dtype
"""
class TrainingConfig(dict):

    """

    """
    def __init__(self, mapping=None, **kwargs):
        
        if mapping is None:
            mapping = DefaultTrainingDict
           
        if kwargs:
            mapping.update({str(key): value for key, value in kwargs.items()})
        super().__init__(mapping)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key:str=None, value:Any=None):
        assert key is not None and value is not None, "key and value must be specified"
        super().__setitem__(key, value)