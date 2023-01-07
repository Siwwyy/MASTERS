
import torch
import torch.nn                             as nn
import torch.nn.functional                  as F
import torchvision.transforms.functional    as tvf
import numpy                                as np

from NeuralNetworks.NN_Base                 import Model_Base
from NeuralNetworks.PreDefined_Blocks       import DoubleConv, DownsampleBlock, UpscaleBlock
from Config.Config                          import TensorType, ShapeType
from typing                                 import Optional, Tuple

class Model_NoCheckerboard(Model_Base):
    """
    UNET architecture based model
    With Upscale layers without ConvTranspose2d

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
        name: str = "Model_NoCheckerboard",
        input_shape: ShapeType = (1, 3, 1920, 1080),
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__(name, input_shape)

        # Uniforms
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Amout of conv features per layer
        divider = 8
        conv_features = np.array([64//divider, 128//divider, 256//divider, 512//divider], dtype=np.int32)

        # Skip connections
        self.skip_connections = [torch.empty((1,1,1,1)), torch.empty((1,1,1,1)), torch.empty((1,1,1,1)), torch.empty((1,1,1,1))]

        # Downsample layers
        self.downsample_block1 = DownsampleBlock(in_channels, conv_features[0])
        self.downsample_block2 = DownsampleBlock(conv_features[0], conv_features[1])
        self.downsample_block3 = DownsampleBlock(conv_features[1], conv_features[2])
        self.downsample_block4 = DownsampleBlock(conv_features[2], conv_features[3])

        # Bottleneck layer
        self.bottleneck = DoubleConv(conv_features[3], conv_features[3] * 2)

        # Upsample layers | conv_features*2 -> because of skip_connection
        self.upsample_block1 = UpscaleBlock(conv_features[3] * 2, conv_features[3])
        self.upsample_block2 = UpscaleBlock(conv_features[2] * 2, conv_features[2])
        self.upsample_block3 = UpscaleBlock(conv_features[1] * 2, conv_features[1])
        self.upsample_block4 = UpscaleBlock(conv_features[0] * 2, conv_features[0])

        # Last upsample conv
        self.upsample_block5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                             nn.Conv2d(conv_features[0], conv_features[0] // 2, kernel_size=1)) #1x1 conv, to accomplish what ConvTranspose2d does, kind of kernel_size
        #self.upsample_block5 = UpscaleBlock(conv_features[0], conv_features[0] // 2)

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
        return self.final_conv(x)


def test():
    x = torch.randn((1, 3, 3840, 2160))
    model = Model_UNET(in_channels=3, out_channels=3)
    preds = model(x)
    assert preds.shape == x.shape

