from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod

from torch.nn.modules import activation
from Config import TensorType, ShapeType, _NNBaseClass
from ImageProcessing.Colorspace import rgbToGrayScale
from ImageProcessing.ImageGradient import calculateGradientMagnitude, _AVAILABLE_KERNELS
from Dataloader import saveEXR

"""
A Reduced-Precision Network for Image Reconstruction (QWNET)
MANU MATHEW THOMAS, University of California, Santa Cruz
KARTHIK VAIDYANATHAN, Intel Corporation
GABOR LIKTOR, Intel Corporation
ANGUS G. FORBES, University of California, Santa Cruz
"""

# KPN Helpers and other stuffs etc.


class InputProcessing(_NNBaseClass):
    """
    InputProcessing
    """

    def __init__(self):
        pass

    def forward(
        self, input: TensorType = None, warpedOutput: TensorType = None
    ) -> TensorType:

        assert (
            input.dim() == 4 and warpedOutput.dim() == 4
        ), f"Forward function params must be in NCHW (dim==4), but got input: {input.dim()} and warpedOutput: {warpedOutput.dim()}"
        grayScaleInput = rgbToGrayScale(input)
        gradMagnitude = calculateGradientMagnitude(
            input, _AVAILABLE_KERNELS["sobelKernel"]
        )

        return torch.cat(
            [grayScaleInput, warpedOutput], dim=-3
        )  # assuming NCHW, concatenate through C channel


class EncoderBlock(_NNBaseClass):
    """
    EncoderBlock
    """

    def __init__(self):
        layers = nn.ModuleDict(
            {
                "conv3x3": nn.Conv2d(
                    3, 3, (3, 3)
                ),  # TODO how many channels are in/out, for now keep 3 as we operate on RGB data
                "batchNorm": nn.BatchNorm2d(3),
                "elu": nn.ELU(),
                "conv3x3": nn.Conv2d(
                    3, 3, (3, 3)
                ),  # TODO how many channels are in/out, for now keep 3 as we operate on RGB data
                "batchNorm": nn.BatchNorm2d(3),
                "elu": nn.ELU(),
                "maxPool2x2": nn.MaxPool2d((2, 2)),
            }
        )

    def forward(self, input: TensorType = None) -> TensorType:
        ...


class DecoderBlock(_NNBaseClass):
    """
    DecoderBlock
    """

    def __init__(self):
        layers = nn.ModuleDict(
            {
                "conv3x3": nn.Conv2d(
                    3, 3, (3, 3)
                ),  # TODO how many channels are in/out, for now keep 3 as we operate on RGB data
                "batchNorm": nn.BatchNorm2d(3),
                "elu": nn.ELU(),
                "conv3x3": nn.Conv2d(
                    3, 3, (3, 3)
                ),  # TODO how many channels are in/out, for now keep 3 as we operate on RGB data
                "batchNorm": nn.BatchNorm2d(3),
                "elu": nn.ELU(),
                "maxPool2x2": nn.MaxPool2d((2, 2)),
            }
        )

    def forward(self, input: TensorType = None) -> TensorType:
        ...


@dataclass
class ModelKPNInputs:
    """
    Dataclass which encapsulates additional input informations to the model
    """

    inputShape: ShapeType = (1, 3, 1920, 1080)  # NCHW
    outputShape: ShapeType = (1, 3, 1920, 1080)  # NCHW


class ModelKPN(_NNBaseClass):
    """
    ModelKPN QWNET
    """

    def __init__(self, name: str = "ModelKPN", modelInputs: ModelKPNInputs = None):
        super().__init__()
        self.name = name
        self.inputShape = modelInputs.inputShape

    def forward(self, x: TensorType = None) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"

    def __repr__(self):
        return self.name
