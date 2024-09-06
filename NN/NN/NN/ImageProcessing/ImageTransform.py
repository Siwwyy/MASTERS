from functools import partial
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Union
from Config.BaseTypes import TensorType

import torch
import torch.nn.functional as F

__all__ = []


def getScreenCoordsFromNDC(
    tensor: TensorType = None, screenDimResolution: int = None
) -> TensorType:
    assert (
        tensor.dim() == 4
    ), "input tensor must be in NCHW format, but got with dim {}".format(tensor.dim())
    assert screenDimResolution is not None, "screenDimResolution must be provided!"
    return ((tensor + 1.0) * screenDimResolution) * 0.5


def getNDCFromScreenCoords(
    tensor: TensorType = None, screenDimResolution: int = None
) -> TensorType:
    assert (
        tensor.dim() == 4
    ), "input tensor must be in NCHW format, but got with dim {}".format(tensor.dim())
    assert screenDimResolution is not None, "screenDimResolution must be provided!"
    return (tensor / screenDimResolution) * 2.0 - 1


def getWorldCoordsFromClip(
    tensor: TensorType = None, screenDimResolution: int = None
) -> TensorType:
    r"""
    see https://carmencincotti.com/2022-05-09/coordinates-from-3d-to-2d/
    """
    assert (
        tensor.dim() == 4
    ), "input tensor must be in NCHW format, but got with dim {}".format(tensor.dim())
    assert screenDimResolution is not None, "screenDimResolution must be provided!"
    return 0.0


def getLinearDepth(depth: TensorType = None) -> TensorType:
    r"""
    fbCoords(n).z=0.5∗(f−n)∗z.y+0.5∗(f+n)
    depth(n) = vp.minDepth + n.z × ( vp.maxDepth - vp.minDepth )
    https://learnopengl.com/Advanced-OpenGL/Depth-testing
     return (2.0 * near * far) / (far + near - z * (far - near));
    """
    near = 0.001
    far = 1000
    return (2.0 * near * far) / (far + near - depth * (far - near))


def getNDCGrid(
    width: int = None, height: int = None, indexing: str = "xy"
) -> TensorType:
    r""" """
    _NDC_MIN_ = -1.0
    _NDC_MAX_ = 1.0
    X_NDC = torch.linspace(
        start=_NDC_MIN_, end=_NDC_MAX_, steps=width, dtype=torch.float16
    )
    Y_NDC = torch.linspace(
        start=_NDC_MIN_, end=_NDC_MAX_, steps=height, dtype=torch.float16
    )
    return torch.meshgrid([X_NDC, Y_NDC], indexing=indexing)


def getScreenCoordGrid(
    width: int = None, height: int = None, indexing: str = "xy"
) -> TensorType:
    r""" """
    X_SCREEN = torch.linspace(start=0, end=width, steps=width, dtype=torch.float16)
    Y_SCREEN = torch.linspace(start=0, end=height, steps=height, dtype=torch.float16)
    return torch.meshgrid([X_SCREEN, Y_SCREEN], indexing=indexing)


import numpy as np


def reproject(
    prevColor: TensorType = None,
    currMV: TensorType = None,
    currDepth: TensorType = None,
    projMat: TensorType = None,
) -> TensorType:

    r"""
    https://matt77hias.github.io/blog/2017/10/19/ndc-to-projection-to-view-space.html
    """

    assert (
        prevColor.dim() == 4 and currMV.dim() == 4 and currDepth.dim() == 4
    ), "Input params must be in NCHW format, but got dims: 1st: {} 2nd: {} 3rd: {}".format(
        prevColor.dim(), currMV.dim(), currDepth.dim()
    )
    torch.set_printoptions(sci_mode=False)

    print(projMat)

    Z_proj = projMat[3, 2]
    W_proj = projMat[2, 2]
    X_proj = 1.0 / projMat[0, 0]  # get inverse of value, 1/x will become x
    Y_proj = 1.0 / projMat[1, 1]  # get inverse of value, 1/y will become y

    xx = np.tile(np.arange(1920, dtype=np.float32)[None, :], (1080, 1))
    yy = np.tile(np.arange(1080, dtype=np.float32)[:, None], (1, 1920))

    xx = 2 * (xx / 1920) - 1
    yy = (2 * (yy / 1080) - 1) * 1

    Z_cam = Z_proj / (currDepth + W_proj)
    X_cam = X_proj * xx * Z_cam
    Y_cam = Y_proj * yy * Z_cam

    # # Get screen offset from NDC
    # XGrid = getScreenCoordsFromNDC(currMV[:, :1, :, :], 1920)
    # YGrid = getScreenCoordsFromNDC(currMV[:, 1:2, :, :], 1080)
    # ZGrid = currDepth

    # # XNDC = XGrid / 1920 * 2.0 - 1.0
    # # YNDC = YGrid / 1080 * 2.0 - 1.0
    # XNDC = getNDCFromScreenCoords(XGrid, 1920)
    # YNDC = getNDCFromScreenCoords(YGrid, 1080)

    return torch.cat([X_cam, Y_cam, Z_cam], dim=-3)
    # return torch.cat([Z_cam], dim=-3)


from Dataloader.DataloaderUtils import loadEXR, loadUnrealCSV, saveEXR

x, y = getNDCGrid(1920, 1080)
print(x[:5][:5])
print(y[:5][:5])
print()
i, j = getNDCGrid(1920, 1080, indexing="ij")
print(i[0][:5])
print(j[0][:5])
# print(abc)

# pthPrevColor = PureWindowsPath(
#     # r"F:\MASTERS\UE4\DATASET\InfiltratorDemo_4_26_2\DumpedBuffers\1920x1080-native\SceneColor\30.exr"
#     r"F:\MASTERS\UE4\DATASET\SubwaySequencer_4_26_2\DumpedBuffers\1920x1080-native\SceneColorTexture\4.exr"
# )

# pthMV = PureWindowsPath(
#     # r"F:\MASTERS\UE4\DATASET\InfiltratorDemo_4_26_2\DumpedBuffers\1920x1080-native\SceneColor\30.exr"
#     r"F:\MASTERS\UE4\DATASET\SubwaySequencer_4_26_2\DumpedBuffers\1920x1080-native\SceneVelocityTexture\5.exr"
# )

# pthDepth = PureWindowsPath(
#     # r"F:\MASTERS\UE4\DATASET\InfiltratorDemo_4_26_2\DumpedBuffers\1920x1080-native\SceneColor\30.exr"
#     r"F:\MASTERS\UE4\DATASET\SubwaySequencer_4_26_2\DumpedBuffers\1920x1080-native\SceneDepthTexture\5.exr"
# )

# prevColorBuffer = loadEXR(str(pthPrevColor))
# mvBuffer = loadEXR(str(pthMV), channels=["R", "G"])
# depthBuffer = loadEXR(str(pthDepth), channels=["R"])


# pth = Path(
#     r"F:\MASTERS\UE4\DATASET\SubwaySequencer_4_26_2\DumpedBuffers\info_Native.csv"
# )
# projMat = loadUnrealCSV(pth, startsWithFilter="Proj_Mat")
# print(projMat)

# reprojected = reproject(
#     prevColorBuffer.unsqueeze(0),
#     mvBuffer.unsqueeze(0),
#     depthBuffer.unsqueeze(0),
#     torch.from_numpy(projMat.values).to(torch.float32)[5].view(4, 4),
# )


# outPth = PureWindowsPath(r"F:\MASTERS\testNDCToCameraSpace.exr")
# saveEXR(str(outPth), reprojected.squeeze(0), channels=["R", "G", "B"])


# # pth = Path(r"F:\MASTERS\UE4\DATASET\SubwaySequencer_4_26_2\DumpedBuffers\info_Native.csv")
# # a = loadUnrealCSV(pth, startsWithFilter="Proj_Mat")
# # print(a)
