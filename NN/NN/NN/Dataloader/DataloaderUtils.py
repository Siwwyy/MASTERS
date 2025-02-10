# EXR Utils (EXR is a base input format for this project)
from functools import partial
import OpenEXR
import Imath
import torch
import numpy as np
import pandas as pd

from dataclasses import dataclass

from torch._prims_common import DeviceLikeType
from Config import TensorType, PathType
from Config.BaseTypes import CurrentDevice

__all__ = [
    "TensorDataTypeFP16",
    "TensorDataTypeFP32",
    "loadEXR",
    "saveEXR",
    "readColorBuffer",
    "readVelocityBuffer",
    "readDepthBuffer",
    "loadUnrealCSV",
]

# Utility dataclasses for encapsulating data types for libs used in project


@dataclass
class TensorDataTypeFP16:
    """Encapsulating fp16 datatype for libs: pytorch, numpy, openEXR"""

    pytorch = torch.float16
    numpy = np.float16
    openEXR = Imath.PixelType(Imath.PixelType.HALF)


@dataclass
class TensorDataTypeFP32:
    """Encapsulating fp32 datatype for libs: pytorch, numpy, openEXR"""

    pytorch = torch.float32
    numpy = np.float32
    openEXR = Imath.PixelType(Imath.PixelType.FLOAT)


def loadEXR(absolutePath: str, channels: list[str] | None = None) -> TensorType:
    r"""
    Loading exr files

    Attributes
    ----------
    absolutePath : str
        absolute path to .exr file
    channels: list[str] (default is ["R", "G", "B"])
        channels to read, e.g., channels=["R", "G", "B"], channels=["R"] etc.

    Returns
    -------
        PyTorch Tensor in CHW format
    """

    # Check if file under given path is correct
    if not OpenEXR.isOpenExrFile(absolutePath):
        raise ValueError(f"Image {absolutePath} is not a correct exr file")

    if channels is None:
        channels = ["R", "G", "B"]

    # Read header etc.
    exrFile = OpenEXR.InputFile(absolutePath)
    dw = exrFile.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Data type case
    dataTypes = TensorDataTypeFP32
    # check type of first channel to read EXR file with correct type
    if list(exrFile.header()["channels"].values())[0].type == Imath.PixelType(
        Imath.PixelType.HALF
    ):
        dataTypes = TensorDataTypeFP16

    # Read data and write into the pytorch tensor
    out_tensor = torch.empty((len(channels), size[1], size[0]), dtype=dataTypes.pytorch)
    for channelIdx, channel in enumerate(channels):
        buffer = np.fromstring(
            exrFile.channel(channel, dataTypes.openEXR), dtype=dataTypes.numpy
        )
        out_tensor[channelIdx, ...] = torch.from_numpy(buffer).view(size[1], size[0])

    return out_tensor


def saveEXR(
    absolutePath: str, tensor: TensorType = None, channels: list[str] | None = None
):
    r"""
    Saving PyTorch tensor with .exr format
    Data type of tensor is propagated to each .exr's file channel's data type

    Attributes
    ----------
    absolutePath : str
        absolute path to .exr file
    tensor: TensorType (torch.tensor)
        CHW shaped pytorch tensor
    channels: List[str] (default is ["R", "G", "B"])
        channels to read, e.g., channels=["R", "G", "B"], channels=["R"] etc.
    """
    assert tensor is not None, "Tensor can't be None!"
    assert tensor.dim() in [
        2,
        3,
    ], f"Tensor dim must be equal 2 or 3!, not dim = {tensor.dim()}"

    if channels is None:
        channels = ["R", "G", "B"]

    outputExrHeader = OpenEXR.Header(tensor.size(-1), tensor.size(-2))
    channel_dtype = (
        Imath.Channel(Imath.PixelType(OpenEXR.HALF))
        if tensor.dtype is torch.float16
        else Imath.Channel(Imath.PixelType(OpenEXR.FLOAT))
    )
    outputExrHeader["channels"] = dict(
        [(channel, channel_dtype) for channel in channels]
    )
    outputExrHeader["compression"] = Imath.Compression(
        Imath.Compression.ZIP_COMPRESSION
    )  # TODO for now, lets use ZIP compression
    output_file = OpenEXR.OutputFile(absolutePath, outputExrHeader)

    # tensor detach(if tensor is in Autograd graph) and convert to numpy array, then to bytes
    output_file_data = dict(
        [
            (channel, data.detach().numpy().tobytes())
            for channel, data in zip(channels, tensor)
        ]
    )
    output_file.writePixels(output_file_data)
    output_file.close()


# pre-defined/specialization of functions
"""
    Functions below are used for pre-defined way of loading buffers
    i.e., we can specify what we should do before/after loading exr data
    like clamping (at depth buffer case, because 0.0 value is treated as inf, too close to camera)
    We can do even tonemap always after loading etc. not-limited ideas. 
    Next case is readability, because we are not forced to specify which channels to load every time.
"""


def Color(absolutePath: str, device: DeviceLikeType = CurrentDevice):
    color = loadEXR(absolutePath=absolutePath, channels=["R", "G", "B"])
    return color.to(device)


def Velocity(absolutePath: str, device: DeviceLikeType = CurrentDevice):
    velocity = loadEXR(absolutePath=absolutePath, channels=["R", "G"])
    return velocity.to(device)


def Depth(absolutePath: str, device: DeviceLikeType = CurrentDevice):
    depth = loadEXR(absolutePath=absolutePath, channels=["R"])
    return depth.clip(min=1e-5).to(
        device
    )  # because 0.0 value is treated as inf, too close to camera


readColorBuffer = partial(Color)
readVelocityBuffer = partial(Velocity)
readDepthBuffer = partial(Depth)


# CSV from UE Loading utils
_UNREAL_CSV_HEADER_FORMAT_: list[str] = [
    "Frame Number",
    "Frame Name",
    "Proj_Mat_rowY_colX",
    "View_Mat_rowY_colX",
    "Inv_Proj_Mat_rowY_colX",
    "Inv_Proj_Mat_rowY_colX",
    "Inv_View_Mat_rowY_colX",
]

from pandas.core.arrays.numeric import NumericDtype


def loadCSV(
    pathToCSVFile: str = None,
    delimiter: str = ",",
    startsWithFilter: str = None,
    useCols: list[str] = None,
    dtype: NumericDtype = None,
):

    # Read csv
    csvFile = pd.read_csv(
        pathToCSVFile,
        header=0,
        delimiter=delimiter,
        usecols=useCols,
        dtype=dtype,
        index_col=False,
    )

    # If we look for something which starts by specified name, it will be filtered
    # and data retunred will be related to filter string
    if startsWithFilter is not None:
        return csvFile.loc[:, csvFile.columns.str.startswith(startsWithFilter)]
    return csvFile


# load csv specialization
# Its done because, loadCSV loads whole csv, we want more flexibility here ;)
def _loadCSV(
    pathToCSVFile: str = None,
    delimiter: str = ",",
    startsWithFilter: str = None,
    useCols: list[str] = None,
    frameIdx: int = None,
    device: DeviceLikeType = CurrentDevice,
):
    assert frameIdx is not None, "frameIdx must be specified"
    loadedCSV = loadCSV(
        pathToCSVFile=pathToCSVFile,
        delimiter=delimiter,
        startsWithFilter=startsWithFilter,
        useCols=useCols,
    )

    loadedCSV = torch.from_numpy(loadedCSV.values)
    loadedCSV = loadedCSV.to(device=device, dtype=torch.float32)
    loadedCSV = loadedCSV[frameIdx]  # get matrix value at specified index
    # we expect 4x4 matrix like in game engines etc. (homogeneous matrix)
    return loadedCSV.view(4, 4)
    # TODO make it more generic, because now, it will load e.g., frame idx and it will make
    # frame_idx.view(4,4), which has no sense


def _loadJitterXYCSV(
    pathToCSVFile: str = None,
    delimiter: str = ",",
    startsWithFilter: str = None,
    useCols: list[str] = None,
    frameIdx: int = None,
    device: DeviceLikeType = CurrentDevice,
):
    assert frameIdx is not None, "frameIdx must be specified"
    loadedCSV = loadCSV(
        pathToCSVFile=pathToCSVFile,
        delimiter=delimiter,
        startsWithFilter=startsWithFilter,
        useCols=useCols,
    )

    loadedCSV = torch.from_numpy(loadedCSV.values)
    loadedCSV = loadedCSV.to(device=device, dtype=torch.float32)
    loadedCSV = loadedCSV[frameIdx]  # get matrix value at specified index
    # we expect 4x4 matrix like in game engines etc. (homogeneous matrix)
    return loadedCSV


# helper specialization (partial)
loadUnrealCSV = partial(_loadCSV)
loadUnrealJitterXYCSV = partial(_loadJitterXYCSV)
