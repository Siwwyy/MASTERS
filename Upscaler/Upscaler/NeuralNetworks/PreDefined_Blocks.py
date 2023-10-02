import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
import numpy as np

from Config.Config import TensorType, ShapeType
from typing import Optional, Tuple


# UNET Based pre-defined blocks
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
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # def init_weights(conv):
        #    if isinstance(conv, nn.Conv2d):
        #        torch.nn.init.kaiming_normal_(conv.weight)

        # self.conv.apply(init_weights)

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

        conv_out = self.conv(x)
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

    def __init__(self, in_channels: int, out_channels: int, abc: int):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            abc, abc, kernel_size=2, stride=2
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


# UNET similar Upsample Block, but with analytic Upscale instead ConvTranspose2D
class UpscaleBlock(nn.Module):
    """
    Upsample Block of UNET

    Attributes
    ----------
    in_channels : int
        Amount of input channels to upsample block
    out_channels : int
        Amount of output channels to upsample block
    scale_factor : int
        Upscale factor, how much output should be upscaled, compared to input
    mode : str
        Mode of upscaling e.g., 'nearest', 'bilinear' etc.
    ----------
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: Tuple[float, float] = (2.0, 2.0),
        mode: str = "nearest",
    ):
        super().__init__()

        self.upscale_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
        )  # 1x1 conv, to accomplish what ConvTranspose2d does, kind of kernel_size
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(
        self, x: TensorType = None, skip_connection: TensorType = None
    ) -> TensorType:
        assert (
            x is not None and skip_connection is not None
        ), "Input tensor X and skip_connection can't be None!"

        x = self.upscale_layer(x)
        # if shape of skip connection is not equal to input tensor, then just resize it
        # It may differ with 2,1 pixels in dim, due to pooling (max pool)
        if skip_connection.shape != x.shape:
            x = tvf.resize(x, size=(skip_connection.size(-2), skip_connection.size(-1)))

        x = torch.cat([skip_connection, x], dim=1)
        return self.conv(x)
