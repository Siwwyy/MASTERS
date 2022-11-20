import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf

from NeuralNetworks.NN_Base import NN_Base
from Config.Config import TensorType, ShapeType
from typing import Optional

import numpy as np


class DoubleConv(nn.Module):
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


class Model_UNET_Tut(NN_Base):
    def __init__(
        self,
        name: str = "Model_UNET",
        input_shape: ShapeType = (1, 3, 1920, 1080),
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__(name, input_shape)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        features = [64, 128, 256, 512]
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            # feature * 2 -> because we will concatenate from Residual
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: TensorType = None) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = tvf.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

    def _generate_architecture(self) -> Optional[nn.Sequential]:
        pass


class Model_UNET(NN_Base):
    """
    A class used to represent an Animal

    Attributes
    ----------
    name : str
        Name of Model
    input_shape : ShapeType (look at Config.py)
        Input shape to the network
    in_channels : str
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._generate_architecture()

    def forward(self, x: TensorType = None) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"

        # Downsample part
        for idx, downsample_layer in enumerate(self.downsample_part):
            x = downsample_layer(x)
            self.skip_connections.append(x)
            x = self.max_pool(x)

        # Bottleneck part (the deepest one)
        x = self.bottleneck(x)

        # Reverse skip_connections, because order of values is from begin to end
        # so from last layer in upsample (which is output)
        # self.skip_connections[:] = torch.flip(self.skip_connections, dims=0)
        self.skip_connections = self.skip_connections[::-1]

        # Upsample part
        # We iterate with step=2, because we have: ConvTranspose2d, DoubleConv, ConvTranspose2d, ...
        for idx in range(0, len(self.upsample_part), 2):
            x = self.upsample_part[idx](x)
            skip_connection = self.skip_connections[idx // 2]

            # Concat part of upsampling
            # TODO tensor's shape differs!! maybe use a resizing?
            if skip_connection.shape != x.shape:
                x = tvf.resize(
                    x, size=(skip_connection.size(-2), skip_connection.size(-1))
                )
            concat_skip_upsample = torch.cat([skip_connection, x], dim=1)

            x = self.upsample_part[idx + 1](concat_skip_upsample)

        return self.final_conv(x)

    def _generate_architecture(self) -> Optional[nn.Sequential]:

        conv_features = np.array([64, 128, 256, 512], dtype=np.int32)

        # Downsample and Upsample part
        self.downsample_part = nn.ModuleList()
        self.upsample_part = nn.ModuleList()

        # Uniform layers
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # self.skip_connections = torch.zeros(4) # for now, we have 4 skip connections
        self.skip_connections = []  # for now, we have 4 skip connections

        # Generate downscample and upsample part
        temp_in_channels = self.in_channels
        for idx, features in enumerate(conv_features):

            # Downsample part
            self.downsample_part.append(DoubleConv(temp_in_channels, features))
            # self.downsample_part.append(self.max_pool)
            temp_in_channels = features

            # Upsample part
            # take elements from end to begin
            conv_feature_reverse = conv_features[(conv_features.size - 1) - idx]
            # features * 2 -> because we will concatenate from downsample Residual, so we will have 2x more channels
            self.upsample_part.append(
                nn.ConvTranspose2d(
                    conv_feature_reverse * 2,
                    conv_feature_reverse,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.upsample_part.append(
                DoubleConv(conv_feature_reverse * 2, conv_feature_reverse)
            )

        self.bottleneck = DoubleConv(conv_features[-1], conv_features[-1] * 2)
        self.final_conv = nn.Conv2d(
            conv_features[0], self.out_channels, kernel_size=1
        )  # 1x1 convolution at the end

        # print(self)
        # print(self)


def test():
    x = torch.randn((1, 3, 1920, 1080))
    # model = Model_UNET_Tut(in_channels=3, out_channels=3)
    # preds = model(x)
    # print(model)

    model1 = Model_UNET(in_channels=3, out_channels=3)
    preds1 = model1(x)
    assert preds1.shape == x.shape
    assert preds1.shape == x.shape
    # assert torch.allclose(preds, preds1), "predictions should be equal!"
    # assert preds.shape == x.shape
    # assert preds.shape == x.shape
    # assert preds1.shape == x.shape
