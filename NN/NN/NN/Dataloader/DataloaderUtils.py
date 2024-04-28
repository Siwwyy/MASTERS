# EXR Utils (EXR is a base input format for this project)
from dataclasses import dataclass
import OpenEXR
import Imath
import torch
import numpy as np

from typing import Tuple, List
from Config import TensorType, PathType

# Utility dataclasses for encapsulating data types for libs used in project
@dataclass
class TensorDataTypeFP32:
    """Encapsulating fp32 datatype for libs: pytorch, numpy, openEXR"""

    pytorch = torch.float32
    numpy = np.float32
    openEXR = Imath.PixelType(Imath.PixelType.FLOAT)


@dataclass
class TensorDataTypeFP16:
    """Encapsulating fp16 datatype for libs: pytorch, numpy, openEXR"""

    pytorch = torch.float16
    numpy = np.float16
    openEXR = Imath.PixelType(Imath.PixelType.HALF)


def loadEXR(absolutePath: str, channels: List[str] | None = None) -> TensorType:
    r"""
    Loading exr files

    Attributes
    ----------
    absolutePath : str
        absolute path to .exr file
    channels: Optional[List[str]] (default is None)
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
    absolutePath: str, tensor: TensorType = None, channels: List[str] | None = None
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
    channels: Optional[List[str]] (default is None)
        channels to read, e.g., channels=["R", "G", "B"], channels=["R"] etc.
    """
    assert tensor is not None, "Tensor can't be None!"
    assert tensor.dim() in [2, 3], "Tensor dim must be equal 2 or 3!"

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
