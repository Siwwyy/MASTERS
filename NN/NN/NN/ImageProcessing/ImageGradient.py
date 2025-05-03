import torch
import torch.nn.functional as F

from functools import partial
from typing import Any, Callable, Union
from NN.Config.BaseTypes import TensorType

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
        NCHW shaped pytorch tensor

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
        NCHW shaped pytorch tensor

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
    return torch.sqrt(kernelOutput[:, :1].pow(2) + kernelOutput[:, 1:2].pow(2))


def calculateGradientDirection(inputTensor: TensorType = None, kernelType: Callable | None = None) -> TensorType:  # type: ignore
    r"""

    Calculate Gradient Direction function

    Attributes
    ----------
    inputTensor: TensorType (torch.tensor)
        NCHW shaped pytorch tensor

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
    return torch.atan2(
        kernelOutput[:, 1:2], kernelOutput[:, :1]
    )  # atan2(y_direction, x_direction)
