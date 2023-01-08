from NeuralNetworks.NN_Base                     import Model_Base
from NeuralNetworks.UNet                        import Model_UNET
from NeuralNetworks.Model_Custom                import Model_Custom
from NeuralNetworks.Model_NoCheckerboard        import Model_NoCheckerboard
from Dataset.Dataset_Base                       import Dataset_Base
from Dataset.Dataset_UE                         import Dataset_UE, FullDataset_UE
from Config.Config                              import CurrentDevice, GetTrainingsPath, GetInferencePath
from Config.TrainingConfig                      import TrainingConfig, ModelHyperparameters


from dataclasses                                import dataclass,astuple
from torch.utils.data                           import DataLoader
from torch                                      import optim
from typing                                     import Any, Dict
from pathlib                                    import Path
from functools                                  import partial


import torch
import torch.nn                                 as nn
import inspect





#Default Configs for objects, pipelines etc. used in project
HyperparametersDict = {
    'className':                ModelHyperparameters,
    'args': {
        'in_channels':          3,
        'out_channels':         3,
        'learning_rate':        0.0001,
        'batch_size':           32,
        'num_epochs':           600
        }
}

TrainDatasetDict = {
    'className':                FullDataset_UE,
    'args': {
        'name':                 'FullDataset_UE',
        'ds_root_path':         Path("E:/MASTERS/UE4/DATASET/"),
        'ue_projects_list':     ["SubwaySequencer_4_26_2", "Rainforest_Scene_4_26_2"],
        'crop_coords':          (900, 1028, 500, 628),
        'transforms':           None,
        'cached':               False
        }
}

ValidDatasetDict = {
    'className':                Dataset_UE,
    'args': {
        'name':                 'Dataset_UE',
        'ds_root_path':         Path("E:/MASTERS/UE4/DATASET/InfiltratorDemo_4_26_2/DumpedBuffers"),
        'csv_root_path':        Path("E:/MASTERS/UE4/DATASET/InfiltratorDemo_4_26_2/DumpedBuffers/info_Native.csv"),
        'crop_coords':          (900, 1028, 500, 628),
        'transforms':           None,
        'cached':               False
        }
}

TrainDataloaderDict = {
    'className':                DataLoader,
    'args': {
        'dataset':              TrainDatasetDict['className'],
        'batch_size':           HyperparametersDict['args']['batch_size'],
        'shuffle':              True,
        'drop_last':            True,
        'pin_memory':           True
        }
}

ValidDataloaderDict = {
    'className':                DataLoader,
    'args': {
        'dataset':              ValidDatasetDict['className'],
        'batch_size':           HyperparametersDict['args']['batch_size'],
        'shuffle':              True,
        'drop_last':            True,
        'pin_memory':           True
        }
}

ModelDict = {
    'className':                Model_NoCheckerboard,
    'args': {
        'in_channels':          HyperparametersDict['args']['in_channels'],
        'out_channels':         HyperparametersDict['args']['out_channels']
        }
}

CriterionDict = {
    'className':                nn.MSELoss,
    'args': {
       'reduction':             'mean' 
       }
}

OptimizerDict = {
    'className':                optim.AdamW,
    'args': {
       'params':                0,
       'lr':                    HyperparametersDict['args']['learning_rate']
        }
}

#Core dict contains paths to folders, dtype used in model, device etc.
CoreDict = {
    'device':                   CurrentDevice,
    'dtype':                    torch.float32,
    #'model_save_path':          GetTrainingsPath(stem=str(ModelDict['className'])+"/epoch{}".format(HyperparametersDict['args']['num_epochs'])),
    #'model_load_path':          GetTrainingsPath(stem=str(ModelDict['className'])+"/epoch{}".format(HyperparametersDict['args']['num_epochs'])),
    #'model_inference_path':     GetInferencePath(stem=str(ModelDict['className'])+"/epoch{}".format(HyperparametersDict['args']['num_epochs']))
}

""" 
    Training, Dataset, Model etc. config, device, dtype utility
"""
class ConfigMapping:

    """
        Config Mapping for simple types, which does not 
        require to be instantiated

        e.g., 
        current_dtype   = torch.float32
        current_device  = torch.device("cpu")
        my_dummy_bool   = True
    """
    def __init__(self, mapping=None, **kwargs):
        if mapping is None:
            mapping = {}
        if kwargs:
            mapping.update({str(key): value for key, value in kwargs.items()})

        for key,value in mapping.items():
            setattr(self, key, value)
           
    def __getitem__(self, key):
        assert hasattr(self, key), f"Class does not have given {key} attribute, current attributes are: {list(self.__dict__.keys())}"
        return getattr(self, key)

    def __setitem__(self, key:str=None, value:Any=None):
        assert key is not None and value is not None, "key and value must be specified"
        assert hasattr(self, key), f"Class does not have given {key} attribute, current attributes are: {list(self.__dict__.keys())}"
        setattr(self, key, value)



class ClassConfigMapping(ConfigMapping):

    """
        Config Mapping for Classes, which should be
        instantiated by a real object

        e.g., 
        current_loss        = nn.MSELoss(...)
        current_optimizer   = optim.Adam(...)
    """
    def __init__(self, mapping=None, **kwargs):
        if mapping is None:
            mapping = {}
        if kwargs:
            mapping.update({str(key): value for key, value in kwargs.items()})

        fullSpec        = inspect.getfullargspec(mapping['className'])
        requiredArgs    = fullSpec.args
        specifiedArgs   = mapping['args']
        requiredArgs.remove('self') #remove key "self" from requiredArgs, because it is not a required arg
        for requiredArg in requiredArgs:
            if requiredArg not in specifiedArgs.keys():
                #If arg is not specified, then set it to None
                mapping[requiredArg] = None
                setattr(self, requiredArg, None)   
                continue

            setattr(self, requiredArg, specifiedArgs[requiredArg])

        self.className = mapping['className']
           
    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key:str=None, value:Any=None):
        return super().__setitem__(key, value)

    def __call__(self):
        dictWithoutClassName = self.__dict__.copy()
        dictWithoutClassName.pop("className")
        return self.className(**dictWithoutClassName)

    #DefaultDict['hyperparams'] =            hyperparams
    #DefaultDict['train_ds'] =               train_ds
    #DefaultDict['train_dataloader'] =       train_loader
    #DefaultDict['valid_dataloader'] =       valid_loader
    #DefaultDict['model'] =                  model
    #DefaultDict['criterion'] =              criterion
    #DefaultDict['optimizer'] =              optimizer
    #DefaultDict['device'] =                 CurrentDevice
    #DefaultDict['dtype'] =                  dtype
    #DefaultDict['model_save_path'] =        GetTrainingsPath(stem=str(model))
    #DefaultDict['model_load_path'] =        GetTrainingsPath(stem=str(model))
    #DefaultDict['model_inference_path'] =   GetInferencePath(stem=str(model))

    #    # Create dataloader for training and validating
    #train_loader = DataLoader(dataset=train_ds, batch_size=hyperparams.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    #valid_loader = DataLoader(dataset=valid_ds, batch_size=hyperparams.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    ### Initialize network
    ##model = Model_Custom(in_channels=hyperparams.in_channels, 
    ##                     out_channels=hyperparams.out_channels).to(device=CurrentDevice, dtype=dtype)
    ## Initialize network
    #model = Model_NoCheckerboard(in_channels=hyperparams.in_channels, 
    #                             out_channels=hyperparams.out_channels).to(device=CurrentDevice, dtype=dtype)

    ## Loss and optimizer
    #criterion = nn.MSELoss()
    #optimizer = optim.AdamW(model.parameters(), lr=hyperparams.learning_rate)

coreCfg                 = ConfigMapping(CoreDict)
train_ds                = ClassConfigMapping(TrainDatasetDict)
valid_ds                = ClassConfigMapping(ValidDatasetDict)
train_loader            = ClassConfigMapping(TrainDataloaderDict)
valid_loader            = ClassConfigMapping(ValidDataloaderDict)
model                   = ClassConfigMapping(ModelDict)
criterion               = ClassConfigMapping(CriterionDict)
optimizer               = ClassConfigMapping(OptimizerDict)

abc = criterion()
print(1)