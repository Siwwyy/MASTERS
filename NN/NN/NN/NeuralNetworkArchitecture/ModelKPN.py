from dataclasses import dataclass
from functools import partial
from typing import Mapping, Union
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

    def __init__(self, layers: Union[nn.ModuleDict, Mapping] | None = None):
        super().__init__()
        self.layers = layers
        if self.layers is None:
            self.layers = nn.ModuleDict(
                {  # no need to add 2x the same key, but added just in case,
                    # for now, these are the same, but in future change conv3x3 to
                    # 1conv3x3, 2conv3x3 etc. if the names are the same
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

        # 1 -> conv3x3 -> batchNorm -> elu
        layerOut = self.layers["conv3x3"](input)
        layerOut = self.layers["batchNorm"](layerOut)
        layerOut = self.layers["elu"](layerOut)
        # 2 -> conv3x3 -> batchNorm -> elu
        layerOut = self.layers["conv3x3"](layerOut)
        layerOut = self.layers["batchNorm"](layerOut)
        layerOut = self.layers["elu"](layerOut)
        skipConnection = layerOut.clone()  # skip connection happens after second ELU
        # 3 -> Max Pool 2x2
        return self.layers["maxPool2x2"](layerOut)  # out


class DecoderBlock(_NNBaseClass):
    """
    DecoderBlock
    """

    def __init__(
        self,
        layers: Union[nn.ModuleDict, Mapping] | None = None,
        upsampleBlockType: str = "nearest",
    ):
        self.layers = layers
        if self.layers is None:
            self.layers = nn.ModuleDict(
                {
                    "conv1x1": nn.Conv2d(
                        3, 3, (1, 1)
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

        self.upsampleLayer = partial(F.upsample, scale_factor=2, mode=upsampleBlockType)

    def forward(
        self, input: TensorType = None, skipConnection: TensorType = None
    ) -> TensorType:

        # upsample Input
        upsampledInput = self.upsampleLayer(input)

        # 1 -> skipConnection + upsampledInput to conv1x1 -> Batch Norm -> ELU
        layerOut = self.layers["conv1x1"](
            upsampledInput + skipConnection
        )  # Add skip connection before going to conv1x1
        layerOut = self.layers["batchNorm"](layerOut)
        layerOut = self.layers["elu"](layerOut)
        # 2 -> conv3x3 -> Batch Norm -> ELU
        layerOut = self.layers["conv3x3"](layerOut)
        layerOut = self.layers["batchNorm"](layerOut)
        return self.layers["elu"](layerOut)  # out


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
