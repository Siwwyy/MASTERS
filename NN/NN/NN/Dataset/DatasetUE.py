from DatasetBase import DatasetBase
from Config import TensorType, PathType
from typing import Optional, Tuple
from collections import namedtuple


class DatasetUE(DatasetBase):
    """
    Dataset for data from Unreal Engine

    Attributes
    ----------
    name : str
        Name of dataset
    datasetRootPath : PathType  (str or Path)
        path to root folder of dataset
    csvPath : Optional[PathType] (str of Path)
        optional path to csv file, if not provided csvPath == datasetRootPath
    cropCoords: Optional[Tuple[int, int, int, int]]
        x_min, y_min, x_max, y_max of the single crop, if not specified crop will be 64x64
    transforms: Optional[torchvision.transforms.Compose]
        optional composed torchvision's tranforms
    ----------
    """

    def __init__(
        self,
        datasetRootPath: PathType = None,
        csvPath: Optional[PathType] = None,
        cropCoords: Optional[Tuple[int, int, int, int]] = None,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        super().__init__(datasetRootPath, transforms)
        self.csvPath = csvPath if csvPath is not None else datasetRootPath

        # Named tuple for coords
        cropCoordsTuple = namedtuple("Coords", ["x_min", "y_min", "x_max", "y_max"])
        self.cropCoords = (
            cropCoordsTuple(*cropCoords)
            if cropCoords is not None
            else cropCoordsTuple(x_min=0, y_min=0, x_max=64, y_max=64)
        )

    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int = None) -> TensorType:
        pass
