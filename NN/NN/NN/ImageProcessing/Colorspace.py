import torch

from typing import Union
from NN.Config.BaseTypes import TensorType

__all__ = [
    "rgbToGrayScale",
    "tonemapReinhard",
    "tonemapReinhard_",
    "detonemapReinhard",
    "detonemapReinhard_",
    "gammaCorrection",
]

# torch.utils.backcompat.broadcast_warning.enabled=True
_EXPOSURE_VALUE = 0.5
_GAMMA_COEFFICIENT = 2.2


def rgbToGrayScale(tensor: TensorType, isHDR: bool = False) -> TensorType:
    r"""
    https://en.wikipedia.org/wiki/Luma_(video)

    Converts rgb to Gray Scale single channel (R) image. Input should be tonemapped and gamma corrected

    Attributes
    ----------
    tensor: TensorType (torch.tensor)
        CHW/NCHW shaped pytorch tensor
    isHDR: bool (default is set to False)

    Returns
    -------
        PyTorch Tensor in 1HW/N1HW shape
    """
    assert (
        tensor.size(-3) == 3
    ), "rgbToGrayScale requires to get tensor with 3 channels (RGB), but got {}".format(
        tensor.size(-3)
    )
    if isHDR:
        pass
    return (
        tensor[..., 0:1, :, :] * 0.2126
        + tensor[..., 1:2, :, :] * 0.7152
        + tensor[..., 2:3, :, :] * 0.0722
    )  # Y709


# Tonemapper HDR -> LDR
def tonemapReinhard(tensor: TensorType) -> TensorType:
    r"""
    https://64.github.io/tonemapping/

    Reinhard Tonemapper. Maps HDR to LDR. Tensor should contain RGB channels (N3HW/3HW shape)

    Attributes
    ----------
    tensor: TensorType (torch.tensor)
        3HW/N3HW shaped pytorch tensor

    Returns
    -------
        LDR PyTorch Tensor in CHW/NCHW shape. By default tensor is clipped to
        min value = 1e-4 to avoid e.g., gradients problem
    """
    newTensor = torch.empty_like(tensor)
    newTensor[:] = tensor / (1.0 + tensor)
    return newTensor.clip(min=1e-4)


# Tonemapper HDR -> LDR
def tonemapReinhard_(tensor: TensorType) -> TensorType:
    r"""
    https://64.github.io/tonemapping/

    Reinhard Tonemapper. Maps HDR to LDR. Inplace equivalent of tonemapReinhard.
    Tensor should contain RGB channels (N3HW/3HW shape)

    Attributes
    ----------
    tensor: TensorType (torch.tensor)
        CHW/NCHW shaped pytorch tensor

    Returns
    -------
        Original LDR PyTorch Tensor in CHW/NCHW shape. By default tensor is clipped to
        min value = 1e-4 to avoid e.g., gradients problem
    """
    tensor[:] = tensor / (1.0 + tensor)
    return tensor.clip(min=1e-4)


# Detonemapper LDR -> HDR
def detonemapReinhard(tensor: TensorType) -> TensorType:
    r"""
    https://64.github.io/tonemapping/

    Reinhard Tonemapper. Maps LDR to HDR.
    Tensor should contain RGB channels (N3HW/3HW shape).

    Attributes
    ----------
    tensor: TensorType (torch.tensor)
        CHW/NCHW shaped pytorch tensor

    Returns
    -------
        HDR PyTorch Tensor in CHW/NCHW shape. By default tensor is clipped to
        min value = 1e-4 to avoid e.g., gradients problem
    """
    newTensor = torch.empty_like(tensor)
    newTensor[:] = tensor / (1.0 - tensor)
    return newTensor.clip(min=1e-4)


# Detonemapper LDR -> HDR
def detonemapReinhard_(tensor: TensorType) -> TensorType:
    r"""
    https://64.github.io/tonemapping/

    Reinhard Tonemapper. Maps LDR to HDR. Inplace equivalent of detonemapReinhard.
    Tensor should contain RGB channels (N3HW/3HW shape).

    Attributes
    ----------
    tensor: TensorType (torch.tensor)
        CHW/NCHW shaped pytorch tensor

    Returns
    -------
        Original HDR PyTorch Tensor in CHW/NCHW shape. By default tensor is clipped to
        min value = 1e-4 to avoid e.g., gradients problem
    """
    tensor[:] = tensor / (1.0 - tensor)
    return tensor.clip(min=1e-4)


# Gamma curve adjustments to get sRGB colorspace
def gammaCorrection(
    tensor: TensorType, gammaCoefficient: float = _GAMMA_COEFFICIENT
) -> TensorType:
    r"""
    https://learnopengl.com/Advanced-Lighting/Gamma-Correction

    Gamma correction of image, used to get sRGB standard from RGB
    Tensor should contain RGB channels (N3HW/3HW shape).

    Attributes
    ----------
    tensor: TensorType (torch.tensor)
        CHW/NCHW shaped pytorch tensor

    Returns
    -------
        Gamma corrected image in CHW/NCHW shape.
    """
    return torch.pow(tensor, gammaCoefficient)
