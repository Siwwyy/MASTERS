from pathlib import PureWindowsPath
from typing import Union
from Config.BaseTypes import TensorType

from Dataloader.DataloaderUtils import loadEXR, saveEXR

import torch
import torch.nn.functional as F

__all__ = ["sobelFilter"]


def sobelFilter(tensor: TensorType) -> TensorType:
    r"""
    https://en.wikipedia.org/wiki/Sobel_operator

    Sobel Filter

    Attributes
    ----------
    tensor: TensorType (torch.tensor)
        CHW/NCHW shaped pytorch tensor

    Returns
    -------
        PyTorch Tensor in 2HW/N2HW shape with Sobel info (Horizontal and Vertical gradient)
    """
    assert (
        tensor.size(-3) == 3
    ), "sobelFilter requires to get tensor with 3 channels (RGB), but got {}".format(
        tensor.size(-3)
    )
    # X axis (horizontal) kernel edge detection
    GxKernel = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).to(
        tensor
    )
    # Y axis (vertical) kernel edge detection
    GyKernel = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).to(
        tensor
    )

    # Get correct weight shape for conv2D i.e., (outChannels, inputChannels, height, width)
    GxyKernel = (
        torch.stack([GxKernel, GyKernel]).unsqueeze_(1).repeat(1, 3, 1, 1)
    )  # Shape(2,3,3,3)
    return F.conv2d(tensor, GxyKernel, bias=None, padding=1).requires_grad_(False)


pth = PureWindowsPath(
    r"F:\MASTERS\UE4\DATASET\InfiltratorDemo_4_26_2\DumpedBuffers\1920x1080-native\SceneColor\30.exr"
)
tens = loadEXR(str(pth))

outputTensor = sobelFilter(tens)

outPth = PureWindowsPath(r"F:\MASTERS\testSobel.exr")
saveEXR(str(outPth), outputTensor, channels=["R", "G"])
