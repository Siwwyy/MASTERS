import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from functools import partial
from typing import Mapping, Union
from torch.nn.modules import activation
from NN.Config import TensorType, ShapeType, _NNBaseClass
from NN.ImageProcessing.Colorspace import rgbToGrayScale
from NN.ImageProcessing.ImageGradient import (
    calculateGradientMagnitude,
    _AVAILABLE_KERNELS,
)

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
        super().__init__()

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
        )  # assuming NCHW, concatenate through C channel i.e., -3/1st


class EncoderBlock(_NNBaseClass):
    """
    EncoderBlock
    """

    def __init__(
        self,
        layers: Union[nn.ModuleDict, Mapping] | None = None,
        inChannels: int = 3,
        outChannels: int = 3,
    ):
        super().__init__()
        self.layers = layers
        if self.layers is None:
            self.layers = nn.ModuleDict(
                {  # no need to add 2x the same key, but added just in case,
                    # for now, these are the same, but in future change conv3x3 to
                    # 1conv3x3, 2conv3x3 etc. if the names are the same
                    "conv3x3_1": nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=inChannels,
                        kernel_size=(3, 3),
                    ),  # TODO how many channels are in/out, for now keep 3 as we operate on RGB data
                    "batchNorm_1": nn.BatchNorm2d(num_features=inChannels),
                    "elu_1": nn.ELU(),
                    "conv3x3_2": nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=outChannels,
                        kernel_size=(3, 3),
                    ),  # TODO how many channels are in/out, for now keep 3 as we operate on RGB data
                    "batchNorm_2": nn.BatchNorm2d(num_features=outChannels),
                    "elu_2": nn.ELU(),
                    "maxPool2x2": nn.MaxPool2d(kernel_size=(2, 2)),
                }
            )

    def forward(self, input: TensorType = None) -> tuple[TensorType, TensorType]:

        # 1 -> conv3x3 -> batchNorm -> elu
        layerOut = self.layers["conv3x3_1"](input)
        layerOut = self.layers["batchNorm_1"](layerOut)
        layerOut = self.layers["elu_1"](layerOut)
        # 2 -> conv3x3 -> batchNorm -> elu
        layerOut = self.layers["conv3x3_2"](layerOut)
        layerOut = self.layers["batchNorm_2"](layerOut)
        layerOut = self.layers["elu_2"](layerOut)
        skipConnection = layerOut.clone()  # skip connection happens after second ELU
        # 3 -> Max Pool 2x2
        return self.layers["maxPool2x2"](layerOut), skipConnection  # out


# Decoder
class DecoderBlock(_NNBaseClass):
    """
    DecoderBlock
    """

    def __init__(
        self,
        layers: Union[nn.ModuleDict, Mapping] | None = None,
        inChannels: int = 3,
        outChannels: int = 3,
        upsampleBlockType: str = "nearest",
    ):
        super().__init__()
        self.layers = layers
        if self.layers is None:
            self.layers = nn.ModuleDict(
                {
                    "conv1x1_1": nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=inChannels,
                        kernel_size=(1, 1),
                    ),  # TODO how many channels are in/out, for now keep 3 as we operate on RGB data
                    "batchNorm_1": nn.BatchNorm2d(num_features=inChannels),
                    "elu_1": nn.ELU(),
                    "conv3x3_2": nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=outChannels,
                        kernel_size=(3, 3),
                    ),  # TODO how many channels are in/out, for now keep 3 as we operate on RGB data
                    "batchNorm_2": nn.BatchNorm2d(num_features=outChannels),
                    "elu_2": nn.ELU(),
                }
            )

        self.upsampleLayer = partial(F.upsample, scale_factor=2, mode=upsampleBlockType)

    def forward(
        self, input: TensorType = None, skipConnection: TensorType = None
    ) -> TensorType:

        # upsample Input
        upsampledInput = self.upsampleLayer(input)

        # 1 -> skipConnection + upsampledInput to conv1x1 -> Batch Norm -> ELU
        layerOut = self.layers["conv1x1_1"](
            upsampledInput + skipConnection
        )  # Add skip connection before going to conv1x1
        layerOut = self.layers["batchNorm_1"](layerOut)
        layerOut = self.layers["elu_1"](layerOut)
        # 2 -> conv3x3 -> Batch Norm -> ELU
        layerOut = self.layers["conv3x3_2"](layerOut)
        layerOut = self.layers["batchNorm_2"](layerOut)
        return self.layers["elu_2"](layerOut)  # out


# Input Filter
class InputFilter(_NNBaseClass):
    """
    Input Filter
    """

    def __init__(self):
        super().__init__()

        self.conv1x1 = nn.Conv2d(
            in_channels=3, out_channels=18, kernel_size=(1, 1)
        )  # 1x1 conv
        self.softmax = nn.Softmax()
        self.avgPool = nn.AvgPool2d(kernel_size=(2, 2))

        # Shape [number_of_filters, input_channels, height, width].
        numberOfFilters = 9
        inputChannels = 3
        height = 3
        width = 3
        self.filter3x3_1 = torch.zeros(
            numberOfFilters, inputChannels, height, width, dtype=torch.float32
        )
        self.filter3x3_2 = torch.zeros(
            numberOfFilters, inputChannels, height, width, dtype=torch.float32
        )

    def forward(
        self,
        input: TensorType = None,
        warpedOutput: TensorType = None,
        inputFeatureExtraction: TensorType = None,
    ) -> tuple[TensorType, TensorType]:

        # Get weights to filters
        filterWeights2x3x3 = self.softmax(
            self.conv1x1(inputFeatureExtraction)
        )  # conv1x1 + softmax

        # TODO shape here will be invalid for sure!
        self.filter3x3_1 = filterWeights2x3x3[:9]  # from [0 to 9)
        self.filter3x3_2 = filterWeights2x3x3[9:]  # from [9 to 17)

        # # Make sure i.e., device, dtype are ok
        # self.filter3x3_1.to(input)
        # self.filter3x3_2.to(input)

        # Filter1 3x3 input
        input = F.conv2d(input, self.filter3x3_1, padding="same")
        # Filter2 3x3 warpedOutput
        warpedOutput = F.conv2d(warpedOutput, self.filter3x3_2, padding="same")

        # Skip connection
        skipConnection = input + warpedOutput

        return self.avgPool(skipConnection.clone()), skipConnection  # out


# Filter
class Filter(_NNBaseClass):
    """
    Filter
    """

    def __init__(self):
        super().__init__()

        self.conv1x1 = nn.Conv2d(
            in_channels=3, out_channels=9, kernel_size=(1, 1)
        )  # 1x1 conv
        self.softmax = nn.Softmax()
        self.avgPool = nn.AvgPool2d(kernel_size=(2, 2))

        # Shape [number_of_filters, input_channels, height, width].
        numberOfFilters = 9
        inputChannels = 3
        height = 3
        width = 3
        self.filter3x3 = torch.zeros(
            numberOfFilters, inputChannels, height, width, dtype=torch.float32
        )

    def forward(
        self, input: TensorType = None, inputFeatureExtraction: TensorType = None
    ) -> tuple[TensorType, TensorType]:

        # Get weights to filters
        filterWeights1x3x3 = self.softmax(
            self.conv1x1(inputFeatureExtraction)
        )  # conv1x1 + softmax
        self.filter3x3 = filterWeights1x3x3  # from [0 to 8)

        # Make sure i.e., device, dtype are ok
        # self.filter3x3.to(input)

        # Filter 3x3
        input = F.conv2d(input, self.filter3x3, padding="same")
        # Skip connection
        skipConnection = input.clone()

        return self.avgPool(input.clone()), skipConnection  # out


# FilterPlusSkip
class FilterPlusSkip(_NNBaseClass):
    """
    Filter + Skip
    """

    def __init__(self, upsampleBlockType: str = "nearest"):
        super().__init__()

        self.conv1x1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=(1, 1)
        )  # 1x1 conv
        self.softmax = nn.Softmax()

        # Shape [number_of_filters, input_channels, height, width].
        numberOfFilters = 9
        inputChannels = 3
        height = 3
        width = 3
        self.filter3x3 = torch.zeros(
            numberOfFilters, inputChannels, height, width, dtype=torch.float32
        )
        self.multiplier = torch.tensor([0.0], dtype=torch.float32)

        self.upsampleLayer = partial(F.upsample, scale_factor=2, mode=upsampleBlockType)

    def forward(
        self,
        input: TensorType = None,
        inputFeatureExtraction: TensorType = None,
        skipConnection: TensorType = None,
    ) -> TensorType:

        # Get weights to filters
        filterWeights1x3x3 = self.softmax(
            self.conv1x1(inputFeatureExtraction)
        )  # conv1x1 + softmax
        self.filter3x3 = filterWeights1x3x3[:9]  # from [0 to 9)
        self.multiplier = filterWeights1x3x3[9]  # from 9th

        # Make sure i.e., device, dtype are ok
        # self.filter3x3.to(input)
        # self.multiplier.to(input)

        # Upsample2x2 + Filter 3x3
        input = self.upsampleLayer(input)
        input = F.conv2d(input, self.filter3x3, padding="same")
        # Skip connection
        skip = input.clone()

        # Multiplication by weight (called multiplier here)
        mulOutput = skipConnection * self.multiplier

        return skip + mulOutput  # Output RGB


# QWNET
@dataclass
class ModelKPNInputs:
    """
    Dataclass which encapsulates additional input informations to the model
    """

    inputShape: ShapeType = (1, 3, 256, 256)  # NCHW
    outputShape: ShapeType = (1, 3, 256, 256)  # NCHW


class ModelKPN(_NNBaseClass):
    """
    ModelKPN QWNET
    """

    def __init__(self, name: str = "ModelKPN", modelInputs: ModelKPNInputs = None):
        super().__init__()

        self.name = name

        ##############################
        # Feature Extraction (U-Net) #
        ##############################

        # Input Processing
        self.inputProcessing = InputProcessing()

        # Encoder Block
        self.encoderBlock1 = EncoderBlock(layers=None, inChannels=8, outChannels=32)
        self.encoderBlock2 = EncoderBlock(layers=None, inChannels=32, outChannels=64)
        self.encoderBlock3 = EncoderBlock(layers=None, inChannels=64, outChannels=96)
        self.encoderBlock4 = EncoderBlock(layers=None, inChannels=96, outChannels=128)
        self.encoderBlock5 = EncoderBlock(layers=None, inChannels=128, outChannels=160)

        # Decoder Block
        self.decoderBlock1 = DecoderBlock(layers=None, inChannels=160, outChannels=128)
        self.decoderBlock2 = DecoderBlock(layers=None, inChannels=128, outChannels=96)
        self.decoderBlock3 = DecoderBlock(layers=None, inChannels=96, outChannels=64)
        self.decoderBlock4 = DecoderBlock(layers=None, inChannels=64, outChannels=32)

        ##############################
        #       Filter Network       #
        ##############################

        # Input Filter
        self.inputFilter = InputFilter()

        # Filter
        self.filter1 = Filter()
        self.filter2 = Filter()
        self.filter3 = Filter()
        self.filter4 = Filter()

        # Filter + Skip
        self.filterPlusSkip1 = FilterPlusSkip()
        self.filterPlusSkip2 = FilterPlusSkip()
        self.filterPlusSkip3 = FilterPlusSkip()
        self.filterPlusSkip4 = FilterPlusSkip()

    def forward(
        self, x: TensorType = None, warped_output: TensorType = None
    ) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"

        # Feature Extractor
        out_fe = self.inputProcessing(x, warped_output)

        #
        out_fe, skip_encoderBlock1 = self.encoderBlock1(out_fe)
        out_fe, skip_encoderBlock2 = self.encoderBlock2(out_fe)
        out_fe, skip_encoderBlock3 = self.encoderBlock3(out_fe)
        out_fe, skip_encoderBlock4 = self.encoderBlock4(out_fe)
        out_fe, act1 = self.encoderBlock5(
            out_fe
        )  # TODO check activations skip-connection

        #
        out_db1 = self.decoderBlock1(out_fe, skip_encoderBlock4)
        out_db2 = self.decoderBlock2(out_db1, skip_encoderBlock3)
        out_db3 = self.decoderBlock3(out_db2, skip_encoderBlock2)
        out_db4 = self.decoderBlock4(out_db3, skip_encoderBlock1)

        # Filter Network
        out_fn, skip_inputFilter = self.inputFilter(x, warped_output, out_db4)

        #
        out_fn, skip_filter1 = self.filter1(out_fn, out_db3)
        out_fn, skip_filter2 = self.filter2(out_fn, out_db2)
        out_fn, skip_filter3 = self.filter3(out_fn, out_db1)
        out_fn, _ = self.filter4(out_fn, act1)

        #
        out_fn = self.filterPlusSkip1(out_fn, out_db1, skip_filter3)
        out_fn = self.filterPlusSkip2(out_fn, out_db2, skip_filter2)
        out_fn = self.filterPlusSkip3(out_fn, out_db3, skip_filter1)
        out_fn = self.filterPlusSkip4(out_fn, out_db4, skip_inputFilter)

        return out_fn  # output RGB

    def __repr__(self):
        return self.name


if __name__ == "__main__":

    model_inputs = ModelKPNInputs()

    # Test
    x = torch.randn((1, 3, 64, 64))
    model = ModelKPN(modelInputs=model_inputs)
    pred = model(x)
    assert pred.shape == x.shape
