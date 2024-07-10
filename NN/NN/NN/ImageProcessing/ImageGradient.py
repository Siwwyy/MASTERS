from functools import partial
from pathlib import PureWindowsPath
from typing import Any, Callable, Union
from Config.BaseTypes import TensorType

import torch
import torch.nn.functional as F

__all__ = [
    "_AVAILABLE_KERNELS",
    "sobelKernel",
    "calculateGradientMagnitude",
    "calculateGradientDirection",
]


def sobelKernel(tensor: TensorType) -> TensorType:
    r"""
    https://en.wikipedia.org/wiki/Sobel_operator

    Sobel Kernel

    Attributes
    ----------
    tensor: TensorType (torch.tensor)
        CHW/NCHW shaped pytorch tensor

    Returns
    -------
        PyTorch Tensor in 2HW/N2HW shape with Sobel info (Horizontal and Vertical gradient)
        Tensor([..., [Gx, Gy], ...])
    """
    assert (
        tensor.size(-3) == 3
    ), "sobelKernel requires to get tensor with 3 channels (RGB), but got {}".format(
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


_AVAILABLE_KERNELS: dict[str, Callable] = {"sobelKernel": partial(sobelKernel)}


def calculateGradientMagnitude(inputTensor: TensorType = None, kernelType: Callable | None = None) -> TensorType:  # type: ignore
    r"""

    Calculate Gradient Magnitude function

    Attributes
    ----------
    inputTensor: TensorType (torch.tensor)
        CHW/NCHW shaped pytorch tensor

    kernelType: Callable
        kernel function, choosen from _AVAILABLE_KERNELS: dict[str, Callable]

    Returns
    -------
        PyTorch Tensor in N1HW shape with gradient magnitude
    """
    assert (
        inputTensor is not None and kernelType is not None
    ), "inputTensor and kernelType can't be None!"

    kernelOutput = kernelType(inputTensor)
    if kernelOutput.dim() == 3:
        kernelOutput.unsqueeze_(0)
    return torch.sqrt(kernelOutput[:, :1].pow(2) + kernelOutput[:, 1:2].pow(2))


def calculateGradientDirection(inputTensor: TensorType = None, kernelType: Callable | None = None) -> TensorType:  # type: ignore
    r"""

    Calculate Gradient Direction function

    Attributes
    ----------
    inputTensor: TensorType (torch.tensor)
        CHW/NCHW shaped pytorch tensor

    kernelType: Callable
        kernel function, choosen from _AVAILABLE_KERNELS: dict[str, Callable]

    Returns
    -------
        PyTorch Tensor in N1HW shape with gradient direction
    """
    assert (
        inputTensor is not None and kernelType is not None
    ), "inputTensor and kernelType can't be None!"

    kernelOutput = kernelType(inputTensor)
    if kernelOutput.dim() == 3:
        kernelOutput.unsqueeze_(0)
    return torch.atan2(kernelOutput[:, 1:2], kernelOutput[:, :1])


# from Dataloader.DataloaderUtils import loadEXR, saveEXR
# from Colorspace import tonemapReinhard

# pth = PureWindowsPath(
#     # r"F:\MASTERS\UE4\DATASET\InfiltratorDemo_4_26_2\DumpedBuffers\1920x1080-native\SceneColor\30.exr"
#     r"F:\MASTERS\UE4\DATASET\Gradient\DumpedBuffers\1920x1080-native\SceneColor\30.exr"
# )
# tens = loadEXR(str(pth))
# tens = tonemapReinhard(tens)
# print(tens.min(), tens.max())
# outputTensor = sobelKernel(tens)
# outputTensor2 = calculateGradientMagnitude(tens, _AVAILABLE_KERNELS["sobelKernel"])
# outputTensor3 = calculateGradientDirection(tens, _AVAILABLE_KERNELS["sobelKernel"])

# outPth = PureWindowsPath(r"F:\MASTERS\testGradientSobelLDR.exr")
# outPth2 = PureWindowsPath(r"F:\MASTERS\testGradientMagnitudeLDR.exr")
# outPth3 = PureWindowsPath(r"F:\MASTERS\testGradientDirectionLDR.exr")
# saveEXR(str(outPth), outputTensor, channels=["R", "G"])
# saveEXR(str(outPth2), outputTensor2.squeeze(0), channels=["R"])
# saveEXR(str(outPth3), outputTensor3.squeeze(0), channels=["R"])
