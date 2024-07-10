from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta, abstractmethod
from Config import TensorType, ShapeType, _NNBaseClass
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
    def __init__(self):
        pass

    def forward(
        self, input: TensorType = None, output: TensorType = None
    ) -> TensorType:

        gradMagnitude = calculateGradientMagnitude(
            input, _AVAILABLE_KERNELS["sobelKernel"]
        )

        print()


"""
Dataclass which encapsulates additional input informations to the model
"""


@dataclass
class ModelKPNInputs:
    inputShape: ShapeType = (1, 3, 1920, 1080)  # NCHW
    outputShape: ShapeType = (1, 3, 1920, 1080)  # NCHW


class ModelKPN(_NNBaseClass):
    """
    ModelKPN QWNET
    """

    def __init__(self, name: str = "Model_Base", modelInputs: ModelKPNInputs = None):
        super().__init__()
        self.name = name
        self.inputShape = modelInputs.inputShape

    def forward(self, x: TensorType = None) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"

    def __repr__(self):
        return self.name
