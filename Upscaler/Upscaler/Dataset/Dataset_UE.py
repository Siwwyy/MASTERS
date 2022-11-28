import torch
import torch.nn as nn
import torchvision.transforms
import numpy as np

# EXR Utils
import OpenEXR
import Imath

# ------------- #

from Config.Config import TensorType, PathType
from Dataset.Dataset_Base import Dataset_Base
from typing import Optional, Tuple, Union
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

    folder_structure = {"lr": Path("1920x1080-native"), "hr": Path("3840x2160-TAA")}

    subfolders_structure = {"lr": Path("SceneColor"), "hr": Path("TemporalAA")}

    channels = ["R", "G", "B"]  # for now, support only RGB, maybe alpha in future

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

        self.lr_folder = (
            ds_root_path
            / Dataset_UE.folder_structure["lr"]
            / Dataset_UE.subfolders_structure["lr"]
        )
        self.hr_folder = (
            ds_root_path
            / Dataset_UE.folder_structure["hr"]
            / Dataset_UE.subfolders_structure["hr"]
        )

    def __len__(self) -> int:
        return len(self.csv_file)

    def __getitem__(self, idx: int = None) -> Tuple[TensorType, TensorType]:
        assert idx is not None, "Index value can't be None! Should be an integer"
        assert idx < self.__len__(), "Index out of bound"

        # Just to check, if tensor is not given as an indice
        # if so, just return it back to scalar (int type)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_idx, file_name = self.csv_file.iloc[0]

        abosolute_lr_path = self.lr_folder / file_name
        abosolute_hr_path = self.hr_folder / file_name

        # TODO WHOLE EXR LOADING UTILITY HAS TO BE MOVED TO SEPARATE FUNCTION e.g., load_exr!
        # lr file
        lr_exr_file = OpenEXR.InputFile(str(abosolute_lr_path))
        lr_dw = lr_exr_file.header()["dataWindow"]
        size = (lr_dw.max.x - lr_dw.min.x + 1, lr_dw.max.y - lr_dw.min.y + 1)

        # Read each channel of exr file
        CHANNEL_BUFFER = np.zeros(
            (len(Dataset_UE.channels), size[0] * size[1]), dtype=np.float16
        )
        for channel_idx, channel_buf in enumerate(
            lr_exr_file.channels(Dataset_UE.channels)
        ):
            # If read channel is in supported channels, write it to main np buffer
            CHANNEL_BUFFER[channel_idx] = np.frombuffer(channel_buf, dtype=np.float16)

        lr_tensor = torch.from_numpy(
            CHANNEL_BUFFER.reshape(len(Dataset_UE.channels), size[0], size[1])
        )

        # hr file
        hr_exr_file = OpenEXR.InputFile(str(abosolute_hr_path))
        hr_dw = hr_exr_file.header()["dataWindow"]
        size = (hr_dw.max.x - hr_dw.min.x + 1, hr_dw.max.y - hr_dw.min.y + 1)

        # Read each channel of exr file
        CHANNEL_BUFFER = np.zeros(
            (len(Dataset_UE.channels), size[0] * size[1]), dtype=np.float16
        )
        for channel_idx, channel_buf in enumerate(
            hr_exr_file.channels(Dataset_UE.channels)
        ):
            # If read channel is in supported channels, write it to main np buffer
            CHANNEL_BUFFER[channel_idx] = np.frombuffer(channel_buf, dtype=np.float16)

        hr_tensor = torch.from_numpy(
            CHANNEL_BUFFER.reshape(len(Dataset_UE.channels), size[0], size[1])
        )

        return (lr_tensor, hr_tensor)  # maybe, return dict?


def test_ds_ue():
    ds = Dataset_UE(
        ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
        csv_root_path=Path(
            "E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"
        ),
    )

    print(len(ds))
    print(ds[5])
