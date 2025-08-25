from NN.Dataset.DatasetBase import DatasetBase
from NN.Config import TensorType, PathType
from collections import namedtuple

import torch
import torchvision


class DatasetUE(DatasetBase):
    """
    Dataset for data from Unreal Engine

    Attributes
    ----------
    name : str
        Name of dataset
    datasetRootPath : PathType  (str or Path)
        path to root folder of dataset
    csvPath : PathType (str of Path)
        optional path to csv file, if not provided csvPath == datasetRootPath
    cropCoords: tuple[int, int, int, int]
        x_min, y_min, x_max, y_max of the single crop, if not specified crop will be 64x64
    transforms: torchvision.transforms.Compose
        optional composed torchvision's tranforms, if not specified, transform == identity transform
    ----------
    """

    def __init__(
        self,
        datasetRootPath: PathType = None,
        csvPath: PathType | None = None,
        cropCoords_lr: tuple[int, int, int, int] | None = None,
        cropCoords_hr: tuple[int, int, int, int] | None = None,
        transforms: torchvision.transforms.Compose | None = None,
    ):
        super().__init__(datasetRootPath, transforms)
        self.csvPath = csvPath if csvPath is not None else datasetRootPath

        # Named tuple for coords
        cropCoordsTuple = namedtuple(
            "Coords",
            ["x_min", "y_min", "x_max", "y_max"],
            defaults=["0", "0", "64", "64"],
        )

        # Cropping
        self.crop_coords_lr = cropCoordsTuple()
        if cropCoords_lr is None:
            self.crop_coords_lr = cropCoordsTuple(cropCoords_lr)
        if cropCoords_hr is None:
            self.cropCoords_hr = cropCoordsTuple(cropCoords_lr)

    def __len__(self) -> int:
        return self.datasetSize

    def __getitem__(self, idx: int = None) -> TensorType:
        super().__getitem__(idx)
        return torch.zeros(1, 64, 64)


if __name__ == "__main__":
    # pass
    abc = DatasetUE("DDD", "DDD")
