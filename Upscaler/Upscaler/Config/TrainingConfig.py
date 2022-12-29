
from NeuralNetworks.NN_Base             import Model_Base
from NeuralNetworks.UNet                import Model_UNET
from NeuralNetworks.Model_Custom        import Model_Custom
from Dataset.Dataset_Base               import Dataset_Base
from Dataset.Dataset_UE                 import Dataset_UE
from Config.Config                      import CurrentDevice, GetTrainingsPath, GetInferencePath

from dataclasses                        import dataclass,astuple
from torch.utils.data                   import DataLoader
from torch                              import optim
from typing                             import Any, Dict
from pathlib                            import Path

import torch
import torch.nn                         as nn



TrainingDictType = Dict[str, Any]

@dataclass()
class ModelHyperparameters:
    in_channels :int = 3
    out_channels :int = 3
    learning_rate :float = 0.001
    batch_size :int = 32
    num_epochs :int = 15

    def __iter__(self):
        return iter(astuple(self))

"""
    Configs
"""
# Default config
def GetDefaultTrainingDict() -> TrainingDictType:
    """Default Training Configuration Dictonary

    Returns
    -------
        Default Training Configuration Dictonary."""

    dtype = torch.float32
    hyperparams = ModelHyperparameters()

    # Create Dataset for training and validating
    train_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
                          csv_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"),
                          #crop width x height == 128x128 (for now)
                          crop_coords=(900, 1028, 500, 628))

    valid_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers"),
                          csv_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers/info_Native.csv"),
                          #crop width x height == 128x128 (for now)
                          crop_coords=(900, 1028, 500, 628))

    # Create dataloader for training and validating
    train_loader = DataLoader(dataset=train_ds, batch_size=hyperparams.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=hyperparams.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    # Initialize network
    model = Model_UNET(in_channels=hyperparams.in_channels, 
                       out_channels=hyperparams.out_channels).to(device=CurrentDevice, dtype=dtype)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

    DefaultDict = {}
    DefaultDict['hyperparams'] =            hyperparams
    DefaultDict['train_ds'] =               train_ds
    DefaultDict['train_dataloader'] =       train_loader
    DefaultDict['valid_dataloader'] =       valid_loader
    DefaultDict['model'] =                  model
    DefaultDict['criterion'] =              criterion
    DefaultDict['optimizer'] =              optimizer
    DefaultDict['device'] =                 CurrentDevice
    DefaultDict['dtype'] =                  dtype
    DefaultDict['model_save_path'] =        GetTrainingsPath(stem=str(model))
    DefaultDict['model_load_path'] =        GetTrainingsPath(stem=str(model))
    DefaultDict['model_inference_path'] =   GetInferencePath(stem=str(model))
    return DefaultDict


""" 
    Training config, device, dtype utility
"""
class TrainingConfig(dict):

    """

    """
    def __init__(self, mapping=None, **kwargs):
        
        if mapping is None:
            mapping = {}
           
        if kwargs:
            mapping.update({str(key): value for key, value in kwargs.items()})
        #assert self.check_required_keys(mapping), "Missing required key in TrainingConfig Dictonary! Look at check_required_keys method"
        super().__init__(mapping)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key:str=None, value:Any=None):
        assert key is not None and value is not None, "key and value must be specified"
        super().__setitem__(key, value)

    def check_required_keys(self, mapping:dict=None) -> bool:
        assert mapping is not None, "mapping param can't be None!"
        required_keys = ['hyperparams', 'train_ds', 'train_dataloader', 
                         'valid_dataloader', 'model', 'criterion',
                         'optimizer', 'device', 'dtype', 'model_save_path', 
                         'model_load_path', 'model_inference_path']
        return all(required_key in mapping.keys() for required_key in required_keys)



#Baseline config
def GetBaselineConfig():
    """Baseline Training Configuration Dictonary

    Returns
    -------
        Baseline Training Configuration Dictonary."""

    dtype = torch.float32
    hyperparams = ModelHyperparameters()
    hyperparams.in_channels         = 3
    hyperparams.out_channels        = 3
    hyperparams.learning_rate       = 0.0001
    hyperparams.batch_size          = 32
    hyperparams.num_epochs          = 600

    # Create Dataset for training and validating
    train_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
                          csv_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"),
                          #crop width x height == 128x128 (for now)
                          crop_coords=(900, 1028, 500, 628), 
                          cached=False)

    valid_ds = Dataset_UE(ds_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers"),
                          csv_root_path=Path("E:/MASTERS/UE4/InfiltratorDemo_4_26_2/DumpedBuffers/info_Native.csv"),
                          #crop width x height == 128x128 (for now)
                          crop_coords=(900, 1028, 500, 628), 
                          cached=False)

    # Create dataloader for training and validating
    train_loader = DataLoader(dataset=train_ds, batch_size=hyperparams.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=hyperparams.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    # Initialize network
    model = Model_Custom(in_channels=hyperparams.in_channels, 
                         out_channels=hyperparams.out_channels).to(device=CurrentDevice, dtype=dtype)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=hyperparams.learning_rate)

    BaselineTrainingCfg = TrainingConfig()
    BaselineTrainingCfg['hyperparams'] =            hyperparams
    BaselineTrainingCfg['train_ds'] =               train_ds
    BaselineTrainingCfg['train_dataloader'] =       train_loader
    BaselineTrainingCfg['valid_dataloader'] =       valid_loader
    BaselineTrainingCfg['model'] =                  model
    BaselineTrainingCfg['criterion'] =              criterion
    BaselineTrainingCfg['optimizer'] =              optimizer
    BaselineTrainingCfg['device'] =                 CurrentDevice
    BaselineTrainingCfg['dtype'] =                  dtype
    #BaselineTrainingCfg['model_save_path'] =        GetTrainingsPath(stem=str(model))
    #BaselineTrainingCfg['model_load_path'] =        GetTrainingsPath(stem=str(model))
    #BaselineTrainingCfg['model_inference_path'] =   GetInferencePath(stem=str(model))
    BaselineTrainingCfg['model_save_path'] =        GetTrainingsPath(stem=str(model)+"/epoch{}".format(hyperparams.num_epochs))
    BaselineTrainingCfg['model_load_path'] =        GetTrainingsPath(stem=str(model)+"/epoch{}".format(hyperparams.num_epochs))
    BaselineTrainingCfg['model_inference_path'] =   GetInferencePath(stem=str(model)+"/epoch{}".format(hyperparams.num_epochs))
    return BaselineTrainingCfg
