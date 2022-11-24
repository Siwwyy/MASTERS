import torch
import torch.nn as nn

from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
from Config.Config import TensorType, ShapeType


class Dataset_Base(torch.Dataset, metaclass=ABCMeta):
    """
    Dataset Base abstract class

    Attributes
    ----------
    name : str
        Name of dataset
    out_channels : int
    ----------
    """

    def __init__(
        self,
        name: str = "Dataset_Base",
    ):
        super().__init__()
        self.name = name

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.__len__.__name__)
        )

    @abstractmethod
    def __getitem__(self, idx: int = None) -> TensorType:
        assert idx is not None, "Index value can't be None! Should be an integer"
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.__getitem__.__name__)
        )
