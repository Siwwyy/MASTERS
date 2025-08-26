import pandas as pd
from NN.Dataset.DatasetBase import DatasetBase
from NN.Config import TensorType, PathType

import torch
import torchvision


class SceneUE:
    """
    Encapsulates Scene properties etc. for single film/project of Unreal Engine
    """

    def __init__(self):
        pass


class DatasetUE(DatasetBase):
    """
    Dataset for data from Unreal Engine

    Attributes
    ----------
    name : str
        Name of dataset
    dataset_root_path : PathType  (str or Path)
        path to root folder of dataset
    csv_path : PathType (str of Path)
        optional path to csv file, if not provided csv_path == dataset_root_path
    transforms: torchvision.transforms.Compose
        optional composed torchvision's tranforms, if not specified, transform == identity transform
    ----------
    """

    def __init__(
        self,
        dataset_root_path: PathType = None,
        transforms: torchvision.transforms.Compose | None = None,
    ):
        super().__init__(dataset_root_path, transforms)

        # load csv
        # assert csv_path is not None, "Csv path is None!"
        # self.csv_path = csv_path
        # self.frame_details_df: pd.DataFrame = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int = None) -> TensorType:
        super().__getitem__(idx)
        return torch.zeros(1, 64, 64)


if __name__ == "__main__":
    # pass
    abc = DatasetUE("DDD", "DDD")
