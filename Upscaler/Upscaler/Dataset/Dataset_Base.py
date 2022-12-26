import torch
import torch.nn as nn
import torchvision.transforms
import pandas as pd


from Config.Config import TensorType, PathType
from torch.utils.data import Dataset
from typing import Optional
from abc import ABCMeta, abstractmethod


# TODO -> Move it maybe to separate file, which will contain only
# transoform utils!
class IdentityTransform(object):
    """
    Identity transform

    tens = IdentityTransform(tens1)
    tens == tens1

    Attributes
    ----------
    name : str
        Name of dataset
    ds_root_path : PathType (Union[str, Path])
        Root path to dataset
    ----------
    """

    def __init__(self):
        pass

    def __call__(self, tens: TensorType = None):
        """
        Parameters
        ----------
        tens : TensorType (Pytorch tensor)
            Input pytorch's tensor with arbitrary shape.
        Returns
        ---------
        Pytorch's tensor, identity of input!
        """
        assert tens is not None, "Input tens cant be none!"
        return tens

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Dataset_Base(torch.utils.data.Dataset, metaclass=ABCMeta):
    """
    Dataset Base abstract class

    Attributes
    ----------
    name : str
        Name of dataset
    ds_root_path : PathType (Union[str, Path])
        Root path to dataset
    ----------
    """

    def __init__(
        self,
        name: str = "Dataset_Base",
        ds_root_path: PathType = None,
        csv_root_path: Optional[PathType] = None,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        assert ds_root_path is not None, "ds_root_path cant be None!"
        super().__init__()

        self.name = name
        self.ds_root_path = ds_root_path

        # Load csv file if path is specified
        if csv_root_path is not None:
            self.csv_file = pd.read_csv(
                csv_root_path, names=("Frame Idx", "Frame name"), index_col=False
            )
            # self.csv_file = pd.read_csv(csv_root_path, index_col=False)

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
        assert idx is not None, "Index value can't be None! Should be an integer"
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.__getitem__.__name__)
        )
