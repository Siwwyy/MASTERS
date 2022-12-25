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



def load_exr_file(absolute_path:str, channels:Optional[List[str]]=None) -> TensorType:
    """
    Loading exr files

    Attributes
    ----------
    absolute_path : str
        absolute path to .exr file
    channels: Optional[List[str]] (default is None)
        channels to read, e.g., channels=["R", "G", "B"], channels=["R"] etc.

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
    out_tensor = torch.empty((len(channels), size[1], size[0]), dtype=Torch_dtype)
    for channel_idx, channel in enumerate(channels):
        buffer = np.fromstring(exr_file.channel(channel, OpenEXR_pixel_type), dtype=Numpy_dtype)
        out_tensor[channel_idx, ...] = torch.from_numpy(buffer).view(size[1], size[0])#[crop_coords[2]:crop_coords[3], crop_coords[0]:crop_coords[1]]

    return out_tensor


def save_exr(absolute_path:str, tensor:TensorType=None, channels:Optional[List[str]]=None): #TODO make this working!
    """
    Saving PyTorch tensor with .exr format
    Data type of tensor is propagated to each .exr's file channel's data type 

    Attributes
    ----------
    absolute_path : str
        absolute path to .exr file
    tensor: TensorType (torch.tensor)
        CHW shaped pytorch tensor
    channels: Optional[List[str]] (default is None)
        channels to read, e.g., channels=["R", "G", "B"], channels=["R"] etc.
    ----------
    """
    assert tensor is not None, "Tensor can't be None!"
    assert tensor.dim() in [2,3], "Tensor dim must be equal 2 or 3!"

    if channels is None:
        channels = ["R", "G", "B"]


    output_exr_header = OpenEXR.Header(tensor.size(-1), tensor.size(-2))
    channel_dtype = Imath.Channel(Imath.PixelType(OpenEXR.HALF)) if tensor.dtype is torch.float16 else Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
    output_exr_header['channels'] = dict([(channel, channel_dtype) for channel in channels])

    output_file = OpenEXR.OutputFile(absolute_path, output_exr_header)

    # tensor detach(if tensor is in Autograd graph) and convert to numpy array, then to bytes
    output_file_data = dict([(channel, data.detach().numpy().tobytes()) for channel, data in zip(channels, tensor)])
    output_file.writePixels(output_file_data)
    output_file.close()

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
     crop_coords: Optional[Tuple[int, int, int, int]]
        x_min, x_max, y_min, y_max of the crop
    transforms: Optional[torchvision.transforms.Compose]
        optional composed torchvision's tranforms
    ----------
    """

    # Folder/subfolder structure must match Unreal's engine outputs!!
    folder_structure = {"lr": Path("1920x1080-native"), "hr": Path("3840x2160-TAA")}
    subfolders_structure = {"lr": Path("SceneColor"), "hr": Path("TemporalAA")}
    channels = ["R", "G", "B"]  # for now, support only RGB, maybe alpha in future

    def __init__(
        self,
        name: str = "Dataset_UE",
        ds_root_path: PathType = None,
        csv_root_path: Optional[PathType] = None,
        crop_coords: Optional[Tuple[int, int, int, int]] = None,
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


        self.crop_coords = crop_coords
        self.crop_coords_hr = None
        if self.crop_coords is not None:
            self.crop_coords_hr = (crop_coords[0] * 2, crop_coords[1] * 2,
                                    crop_coords[2] * 2, crop_coords[3] * 2)
            #self.crop_coords_hr = *(self.crop_coords[:] * 2,)

        if self.crop_coords is None:
            # x_min, x_max, y_min, y_max
            self.crop_coords = (None, None, None, None)
            self.crop_coords_hr = (None, None, None, None)


    def __len__(self) -> int:
        #return len(self.csv_file)
        return 64
        #return 2

    def __getitem__(self, idx: int = None) -> Tuple[TensorType, TensorType]:
        assert idx is not None, "Index value can't be None! Should be an integer"
        assert idx < self.__len__(), "Index out of bound"

        ## Just to check, if tensor is not given as an indice
        ## if so, just return it back to scalar (int type)
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        file_idx, file_name = self.csv_file.iloc[idx]

        abosolute_lr_path = self.lr_folder / file_name
        abosolute_hr_path = self.hr_folder / file_name

        # lr file
        lr_tensor = load_exr_file(str(abosolute_lr_path), Dataset_UE.channels)[..., self.crop_coords[2]:self.crop_coords[3], self.crop_coords[0]:self.crop_coords[1]]

        # hr file
        hr_tensor = load_exr_file(str(abosolute_hr_path), Dataset_UE.channels)[..., self.crop_coords_hr[2]:self.crop_coords_hr[3], self.crop_coords_hr[0]:self.crop_coords_hr[1]]

        # TODO, add pytorch transforms if needed
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
    save_exr("E:\MASTERS\lr.exr", lr)
    save_exr("E:\MASTERS\hr.exr", hr)

    lr_loaded_tens = load_exr_file("E:\MASTERS\lr.exr", channels=["R","G","B"])

    assert torch.allclose(lr, lr_loaded_tens), "Tensor should be equal after loading -> saving -> loading operations"

    import matplotlib.pyplot as plt
    # Plotting part
    figure = plt.figure(figsize=(15, 20))
    lr = lr * 5.
    hr = hr * 5.
    plt.imshow(hr.permute(1,2,0).to(dtype=torch.float32).cpu().detach().numpy())
    plt.show()



    #####################################
    # cropped images loading check
    ds = Dataset_UE(
        ds_root_path=Path("E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers"),
        csv_root_path=Path(
            "E:/MASTERS/UE4/SubwaySequencer_4_26/DumpedBuffers/info_Native.csv"
        ),
        crop_coords=(900, 964, 500, 564)
    )

    lr, hr = ds[5]
    assert lr is not None and hr is not None, "Loaded EXR images can't be none!"
