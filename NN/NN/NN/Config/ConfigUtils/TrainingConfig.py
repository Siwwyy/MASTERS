import torch


from dataclasses import astuple, dataclass
from torch.utils.data import DataLoader

from Dataset.DatasetBase import DatasetBase
from Loss.LossBase import LossBase
from NeuralNetworkArchitecture.ModelBase import ModelBase


@dataclass
class _ParamsBase:
    def __init_subclass__(cls) -> None:
        pass


@dataclass
class ModelHyperparameters(_ParamsBase):
    r"""Encapsulating model hyperaparameters for training dispatch"""
    inChannels: int = 3
    outChannels: int = 3
    learningRate: float = 0.001
    batchSize: int = 32
    numEpochs: int = 15

    def __dict__(self):
        return {
            "inChannels": self.inChannels,
            "outChannels": self.outChannels,
            "learningRate": self.learningRate,
            "batchSize": self.batchSize,
            "numEpochs": self.numEpochs,
        }

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class DispatchParams(_ParamsBase):
    """Encapsulates parameters of dispatched training e.g., model, loss etc."""

    hyperparams: ModelHyperparameters = None
    dataset: DatasetBase = None
    dataloader: DataLoader = None
    loss: LossBase = None
    optimizer: torch.optim.Optimizer = None
    model: ModelBase = None

    def __dict__(self):
        return {
            "hyperparams": self.hyperparams,
            "dataset": self.dataset,
            "dataloader": self.dataloader,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "model": self.model,
        }

    def __repr__(self):
        return f"""Hyperparams {self.hyperparams} |
                 Dataset {self.dataset} |
                 Dataloader {self.dataloader} |
                 Loss {self.loss} |
                 Optimizer {self.optimizer} |
                 Model {self.model} |"""
