import torch
import torch.nn as nn
import torchvision.transforms

from Config.Config import TensorType, PathType
from Dataset.Dataset_Base import Dataset_Base
from typing import Optional
from pathlib import Path


class Dataset_UE(Dataset_Base):
    """
    Dataset for data from Unreal Engine

    Attributes
    ----------
    name : str
        Name of dataset
    ds_root_path : PathType  (str of Path)
        path to root folder of dataset
    csv_root_path : Optional[PathType] (str of Path)
        optional path to csv file
    transforms: Optional[torchvision.transforms.Compose]
        optional composed torchvision's tranforms
    ----------
    """

    def __init__(
        self,
        name: str = "Dataset_UE",
        ds_root_path: PathType = None,
        csv_root_path: Optional[PathType] = None,
        transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        assert (
            csv_root_path is not None
        ), "Unreal Engine based dataset must contain csv file!"
        super().__init__(name, ds_root_path, csv_root_path, transforms)

        self.lr_folder = ds_root_path / Path("1920x1080-native")
        self.hr_folder = ds_root_path / Path("3840x2160-TAA")

    def __len__(self) -> int:
        return len(self.csv_file)

    def __getitem__(self, idx: int = None) -> TensorType:
        assert idx is not None, "Index value can't be None! Should be an integer"
        assert idx < self.__len__(), "Index out of bound"

        # Just to check, if tensor is not given as an indice
        # if so, just return it back to scalar (int type)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        print(self.csv_file)
        print(self.csv_file.iloc[0])

        # for file_id, file_name in enumerate(Path.iterdir(self.root_dir.absolute())):
        #    if file_id == idx:
        #        image = torch.from_numpy(imread(file_name)).permute(2,0,1).float()
        #        sample = {'image': image}
        #        if self.transform:
        #            sample = self.transform(sample)
        #        return sample
        # return None


def test_ds_ue():
    ds = Dataset_UE(
        ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
        csv_root_path=Path(
            "E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"
        ),
    )

    print(len(ds))
    print(ds[5])
