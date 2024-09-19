from functools import partial
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Union

from torch.nn.modules import padding
from Config.BaseTypes import TensorType

import torch
import torch.nn.functional as F

from Open3D.Visualization import showPointCloud3D

__all__ = [
    "getPixelFromNDC",
    "getNDCFromPixel",
    "getPixelGridFromNDCGrid",
    "getNDCGridFromPixelGrid",
    "getWorldFromClip",
    "getLinearDepth",
    "getNDCGrid",
    "getPixelGrid",
    "projectionToViewSpace",
    "reproject",
]


def getPixelFromNDC(
    tensor: TensorType = None, screenDimResolution: int = None
) -> TensorType:
    assert screenDimResolution is not None, "screenDimResolution must be provided!"
    return (tensor + 1.0) * (screenDimResolution - 1.0) * 0.5


def getNDCFromPixel(
    tensor: TensorType = None, screenDimResolution: int = None
) -> TensorType:
    assert screenDimResolution is not None, "screenDimResolution must be provided!"
    return (tensor / screenDimResolution) * 2.0 - 1.0


def getPixelGridFromNDCGrid(ndcXYGrid: TensorType = None) -> TensorType:
    r"""
    see https://learn.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-rasterizer-stage-getting-started?redirectedfrom=MSDN
        https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-coordinates

        in dx12, bottom-left corner is -1,-1 NDC, but in pytorch manner, its top-left corner, thats why we do not have to invert Y (flip) axis

    """
    ndcXYGrid = ndcXYGrid.clone()  # clone is to avoid using the same memory
    H, W = ndcXYGrid.size(-3), ndcXYGrid.size(-2)
    ndcXYGrid[..., 0:1] = (ndcXYGrid[..., 0:1] + 1.0) * (W - 1.0) * 0.5  # x dir
    ndcXYGrid[..., 1:2] = (ndcXYGrid[..., 1:2] + 1.0) * (H - 1.0) * 0.5  # y dir
    return ndcXYGrid  # now it becames a pixel coords grid


def getNDCGridFromPixelGrid(pixelXYGrid: TensorType = None) -> TensorType:
    r"""
    see https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-coordinates

    """
    pixelXYGrid = pixelXYGrid.clone()  # clone is to avoid using the same memory
    H, W = pixelXYGrid.size(-3), pixelXYGrid.size(-2)
    pixelXYGrid[..., 0:1] = (pixelXYGrid[..., 0:1] / W) * 2.0 - 1.0  # x dir
    pixelXYGrid[..., 1:2] = (pixelXYGrid[..., 1:2] / H) * 2.0 - 1.0  # y dir
    return pixelXYGrid.clone()  # now it becames a ndc coords grid


def getCameraFromClip(
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
    near = 0.0001
    far = 10000
    return (2.0 * near * far) / (far + near - depth * (far - near))


def getNDCGrid(
    width: int = None, height: int = None, indexing: str = "xy"
) -> TensorType:
    r""" """
    _NDC_MIN_ = -1.0
    _NDC_MAX_ = 1.0
    X_NDC = torch.linspace(
        start=_NDC_MIN_, end=_NDC_MAX_, steps=width, dtype=torch.float32
    )  # grid sampler is not implemented for half float16!
    Y_NDC = torch.linspace(
        start=_NDC_MIN_, end=_NDC_MAX_, steps=height, dtype=torch.float32
    )  # grid sampler is not implemented for half float16!
    return torch.meshgrid([X_NDC, Y_NDC], indexing=indexing)


def getPixelGrid(
    width: int = None, height: int = None, indexing: str = "xy"
) -> TensorType:
    r""" """
    X_PIXEL = torch.arange(
        start=0, end=width, step=1, dtype=torch.float32
    )  # grid sampler is not implemented for half float16!
    Y_PIXEL = torch.arange(
        start=0, end=height, step=1, dtype=torch.float32
    )  # grid sampler is not implemented for half float16!
    return torch.meshgrid([X_PIXEL, Y_PIXEL], indexing=indexing)


def projectionToViewSpace(
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

    # TODO -> add max/min values of depth buffer (clamp)

    # fill values from projection matrix at frame X
    Z_proj = projMat[3, 2]
    W_proj = projMat[2, 2]  # w coordinate aka "perspective divisior"
    X_proj = 1.0 / projMat[0, 0]  # get inverse of value, 1/x will become x
    Y_proj = 1.0 / projMat[1, 1]  # get inverse of value, 1/y will become y

    # get NDC coords grid in X,Y axis
    X_NDC_GRID, Y_NDC_GRID = getNDCGrid(1920, 1080)

    # obtain pixels relative to camera position
    Z_cam = Z_proj / (currDepth + W_proj)
    X_cam = X_proj * X_NDC_GRID * Z_cam  # x_proj * x_ndc * z_cam
    Y_cam = Y_proj * Y_NDC_GRID * Z_cam  # y_proj * y_ndc * z_cam

    # X_Pixel_Coords = getPixelFromNDC(X_NDC_GRID, 1920)
    # Y_Pixel_Coords = getPixelFromNDC(Y_NDC_GRID, 1080)

    return torch.cat([X_cam, Y_cam, Z_cam], dim=-3)


def reproject(
    prevColor: TensorType = None,
    currMV: TensorType = None,
    currDepth: TensorType = None,
) -> TensorType:

    r"""
    Reprojection/Warping function
    """

    assert (
        prevColor.dim() == 4 and currMV.dim() == 4 and currDepth.dim() == 4
    ), "Input params must be in NCHW format, but got dims: 1st: {} 2nd: {} 3rd: {}".format(
        prevColor.dim(), currMV.dim(), currDepth.dim()
    )

    # Helpers
    HEIGHT, WIDTH = prevColor.size(-2), prevColor.size(-1)
    prevColorDtype = prevColor.dtype

    # Get grid
    NDC_GRID_X, NDC_GRID_Y = getNDCGrid(WIDTH, HEIGHT)
    NDC_GRID_XY = torch.stack([NDC_GRID_X, NDC_GRID_Y], dim=-1).unsqueeze(
        0
    )  # get batch dim with unsqueeze

    # Permute MV (Velocity) buffer channels to NHWC (otherwise, we are unable to subtract by grid simply)
    nhwc_mv = currMV.permute(0, 2, 3, 1)

    # "Projects" current grid values to previous by subtraction
    NEW_NDC_COORDS = NDC_GRID_XY - nhwc_mv

    return F.grid_sample(
        prevColor.float(),
        NEW_NDC_COORDS,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).to(dtype=prevColorDtype)


# from Dataloader.DataloaderUtils import loadEXR, loadUnrealCSV, readColorBuffer, readDepthBuffer, readVelocityBuffer, saveEXR


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

# prevColorBuffer = readColorBuffer(str(pthPrevColor))
# mvBuffer        = readVelocityBuffer(str(pthMV))
# depthBuffer     = readDepthBuffer(str(pthDepth))


# pthMatProj = Path(
#     r"F:\MASTERS\UE4\DATASET\SubwaySequencer_4_26_2\DumpedBuffers\info_Native.csv"
# )
# # projMat = loadUnrealCSV(pth, startsWithFilter="Proj_Mat")
# # print(projMat)
# projMat = loadUnrealCSV(pthMatProj, startsWithFilter="Proj_Mat", frameIdx=5)

# reprojected = reproject(
#     prevColorBuffer.unsqueeze(0),
#     mvBuffer.unsqueeze(0),
#     depthBuffer.unsqueeze(0)
# )

# # showPointCloud3D(reprojected.squeeze(0), min_range=0.0, max_range=1000.0)
# outPth = PureWindowsPath(r"F:\MASTERS\testReprojection.exr")
# saveEXR(str(outPth), reprojected.squeeze(0), channels=["R", "G", "B"])


# # pth = Path(r"F:\MASTERS\UE4\DATASET\SubwaySequencer_4_26_2\DumpedBuffers\info_Native.csv")
# # a = loadUnrealCSV(pth, startsWithFilter="Proj_Mat")
# # print(a)
