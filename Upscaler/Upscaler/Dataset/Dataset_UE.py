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
from typing import Optional, Tuple, Union, List
from pathlib import Path



def load_exr_file(absolute_path:str, channels:Optional[List[str]]=None, crop_coords:Optional[Tuple[int, int, int, int]]=None) -> TensorType:
    """
    Loading exr files

    Attributes
    ----------
    absolute_path : PathType (Union[Path,str])
        absolute path to .exr file
    channels: Optional[List[str]] (default is None)
        channels to read, e.g., channels=["R", "G", "B"], channels=["R"] etc.
    crop_coords: Optional[Tuple[int, int, int, int]]
        x_min, x_max, y_min, y_max
    ----------
    """

    # Check if file under given path is correct
    if not OpenEXR.isOpenExrFile(absolute_path):
        raise ValueError(f'Image {absolute_path} is not a correct exr file')

    if channels is None:
        channels = ["R", "G", "B"]

    # Read header etc.
    exr_file = OpenEXR.InputFile(absolute_path)
    dw = exr_file.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    if crop_coords is None:
        crop_coords = (0, size[0], 0, size[1])

    # Data type case
    OpenEXR_pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    Numpy_dtype = np.float32
    Torch_dtype = torch.float32
    # check type of first channel to read EXR file with correct type
    if list(exr_file.header()['channels'].values())[0].type == Imath.PixelType(Imath.PixelType.HALF):
        OpenEXR_pixel_type = Imath.PixelType(Imath.PixelType.HALF)
        Numpy_dtype = np.float16
        Torch_dtype = torch.float16

    # Read data and write into the pytorch tensor
    # (num_channels, x_max - x_min, y_max - y_min) -> final tensor shape
    out_tensor = torch.empty((len(channels), crop_coords[3] - crop_coords[2], crop_coords[1] - crop_coords[0]), dtype=Torch_dtype)
    for channel_idx, channel in enumerate(channels):
        buffer = np.fromstring(exr_file.channel(channel, OpenEXR_pixel_type), dtype=Numpy_dtype)
        out_tensor[channel_idx, ...] = torch.from_numpy(buffer.reshape(size[1], size[0]))[crop_coords[2]:crop_coords[3], crop_coords[0]:crop_coords[1]]

    return out_tensor


def save_exr(absolute_path:Path, tensor:TensorType=None, channels:Optional[List[str]]=None): #TODO make this working!
    """
    Loading exr files

    Attributes
    ----------
    absolute_path : PathType (Union[Path,str])
        absolute path to .exr file
    dtype: str
        data type of read exr, can be float16 or float32
    channels: Optional[List[str]] (default is None)
        channels to read, e.g., channels=["R", "G", "B"], channels=["R"] etc.
    ----------
    """
    assert tensor is not None, "Tensor can't be None!"

    if channels is None:
        channels = ["R", "G", "B"]


    hdr = OpenEXR.Header(w, h)
    chan = Imath.Channel(Imath.PixelType(OpenEXR.HALF))
    hdr['channels'] = {'R' : chan, 'G' : chan, 'B' : chan, 'A' : chan}
    x = OpenEXR.OutputFile(absolute_path, hdr)
    x.writePixels({'R': data, 'G': data, 'B': data, 'A' : data})
    x.close()
    #hdr = OpenEXR.Header(100, 100)
    #for chans in [ set("a"), set(['foo', 'bar']), set("abcdefghijklmnopqstuvwxyz") ]:
    #    hdr['channels'] = dict([(nm, Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))) for nm in chans])
    #    x = OpenEXR.OutputFile(absolute_path, hdr)
    #    data = array('f', [0] * (100 * 100)).tobytes()
    #    x.writePixels(dict([(nm, data) for nm in chans]))
    #    x.close()

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
        #return len(self.csv_file)
        return 10

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
        lr_tensor = load_exr_file(str(abosolute_lr_path), channels)

        # hr file
        hr_tensor = load_exr_file(str(abosolute_hr_path), channels)

        return (lr_tensor, hr_tensor)  # maybe, return dict?


def test_ds_ue():
    ds = Dataset_UE(
        ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
        csv_root_path=Path(
            "E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"
        ),
    )

    lr, hr = ds[5]
    assert lr is not None and hr is not None, "Loaded EXR images can't be none!"

    # fix exr loading, NaNs!!!!!!!
    import matplotlib.pyplot as plt
    # Plotting part
    figure = plt.figure(figsize=(15, 20))
    lr = lr * 5.
    hr = hr * 5.
    plt.imshow(lr.permute(1,2,0).to(dtype=torch.float32).cpu().detach().numpy())
    plt.show()
