import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf

from NeuralNetworks.NN_Base import NN_Base
from Config.Config import TensorType, ShapeType
from typing import Optional, Tuple

import numpy as np


class DoubleConv(nn.Module):
    """
    Double Conv Block of UNET

    Attributes
    ----------
    in_channels : int
        Amount of input channels to downsample block
    out_channels : int
        Amount of output channels to downsample block
    ----------
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: TensorType = None) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"
        return self.conv(x)


class DownsampleBlock(nn.Module):
    """
    Downsample Block of UNET

    Attributes
    ----------
    in_channels : int
        Amount of input channels to downsample block
    out_channels : int
        Amount of output channels to downsample block
    pool_layer: Optional[nn.Module]
        Optional pooling layer for block, if not specified == nn.MaxPool2d(kernel_size=(2, 2), stride=2)
    ----------
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = pool_layer
        if pool_layer is None:
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x: TensorType = None) -> Tuple[TensorType, TensorType]:
        assert x is not None, "Input tensor X can't be None!"

        conv_out = self.conv(x).clone()
        return self.pool(conv_out), conv_out


class UpsampleBlock(nn.Module):
    """
    Upsample Block of UNET

    Attributes
    ----------
    in_channels : int
        Amount of input channels to upsample block
    out_channels : int
        Amount of output channels to upsample block
    ----------
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(
        self, x: TensorType = None, skip_connection: TensorType = None
    ) -> TensorType:
        assert (
            x is not None and skip_connection is not None
        ), "Input tensor X and skip_connection can't be None!"

        x = self.conv_transpose2d(x)
        # if shape of skip connection is not equal to input tensor, then just resize it
        # It may differ with 2,1 pixels in dim, due to pooling (max pool)
        if skip_connection.shape != x.shape:
            x = tvf.resize(x, size=(skip_connection.size(-2), skip_connection.size(-1)))

        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)


class Model_UNET(NN_Base):
    """
    UNET architecture based model

    Attributes
    ----------
    name : str
        Name of Model
    input_shape : ShapeType (look at Config.py)
        Input shape to the network
    in_channels : int
        Number of input channels to the network
    out_channels : int
        Number of output channels to the network
    ----------
    """

    def __init__(
        self,
        name: str = "Model_UNET",
        input_shape: ShapeType = (1, 3, 1920, 1080),
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__(name, input_shape)

        # Uniforms
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Amout of conv features per layer
        conv_features = np.array([64, 128, 256, 512], dtype=np.int32)

        # Skip connections
        self.skip_connections = [torch.empty((1,1,1,1)), torch.empty((1,1,1,1)), torch.empty((1,1,1,1)), torch.empty((1,1,1,1))]

        # Downsample layers
        self.downsample_block1 = DownsampleBlock(in_channels, conv_features[0])
        self.downsample_block2 = DownsampleBlock(conv_features[0], conv_features[1])
        self.downsample_block3 = DownsampleBlock(conv_features[1], conv_features[2])
        self.downsample_block4 = DownsampleBlock(conv_features[2], conv_features[3])

        # Bottleneck layer
        self.bottleneck = DoubleConv(conv_features[3], conv_features[3] * 2)

        # Upsample layers
        self.upsample_block1 = UpsampleBlock(conv_features[3] * 2, conv_features[3])
        self.upsample_block2 = UpsampleBlock(conv_features[2] * 2, conv_features[2])
        self.upsample_block3 = UpsampleBlock(conv_features[1] * 2, conv_features[1])
        self.upsample_block4 = UpsampleBlock(conv_features[0] * 2, conv_features[0])

        # Last upsample conv
        #self.upsample_block5 = UpsampleBlock(conv_features[0], conv_features[0] // 2)
        self.upsample_block5 = nn.ConvTranspose2d(
            conv_features[0], conv_features[0] // 2, kernel_size=2, stride=2
        )

        # Final convolution
        self.final_conv = nn.Conv2d(
            conv_features[0] // 2, self.out_channels, kernel_size=1
        )  # 1x1 convolution at the end

    def forward(self, x: TensorType = None) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"

        # Downsample
        for idx in [1, 2, 3, 4]:
            downsample_block = getattr(self, "downsample_block{}".format(idx))
            x, x1 = downsample_block(x)
            self.skip_connections[idx - 1] = x1

        # Bottleneck
        x = self.bottleneck(x).clone()

        # Upsample
        for idx in [1, 2, 3, 4]:
            upsample_block = getattr(self, "upsample_block{}".format(idx))
            x = upsample_block(
                x, self.skip_connections[len(self.skip_connections) - idx]
            )
        

        x = self.upsample_block5(x)
        # Final, last conv
        # TODO Add next conv, to reach upsampling. Right now: out shape == in shape. Should be -> 2x upsaling
        return self.final_conv(x)

    def _generate_architecture(self) -> Optional[nn.Sequential]:
        pass


def test():
    x = torch.randn((1, 3, 3840, 2160))
    model = Model_UNET(in_channels=3, out_channels=3)
    preds = model(x)
    assert preds.shape == x.shape
