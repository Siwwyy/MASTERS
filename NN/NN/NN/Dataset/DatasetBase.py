import torch
import torch.nn as nn
import torchvision.transforms
import pandas as pd


from Config import TensorType, PathType
from Transforms.Transforms import IdentityTransform
from torch.utils.data import Dataset
from typing import Optional
from abc import ABCMeta, abstractmethod


class DatasetBase(torch.utils.data.Dataset, metaclass=ABCMeta):
    r"""
    Dataset Base abstract class

    Attributes
    ----------
    datasetRootPath : PathType (Union[str, Path])
        Root path to dataset
    transforms: torchvision.transforms.Compose
        optional composed torchvision's tranforms, if not specified, transform == identity transform
    ----------
    """

    def __init__(
        self,
        datasetRootPath: PathType = None,
        transforms: torchvision.transforms.Compose | None = None,
    ):
        assert datasetRootPath is not None, "Dataset root path can't be None!"
        super().__init__()

        self.datasetRootPath = datasetRootPath

        # if tranforms has been not specified, then use Identity transform
        # x == id(x)
        if transforms is None:
            self.transforms = IdentityTransform()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.__len__.__name__)
        )

    @abstractmethod
    def __getitem__(self, idx: int = None) -> TensorType:
        assert idx is not None, "Index value can't be None!"
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.__getitem__.__name__)
        )