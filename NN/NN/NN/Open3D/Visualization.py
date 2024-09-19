from typing import List
from Config import TensorType, PathType

import torch
import open3d as o3d
import numpy as np

__all__ = ["showPointCloud3D"]


def showPointCloud2D(
    tensor: TensorType = None, min_range: int = -1000, max_range: int = 1000
) -> None:
    r"""

    Show points cloud in 3D

    Attributes
    ----------
    tensor: TensorType (torch.tensor)
        CHW shaped pytorch tensor

    min_range: int
        minimal range of drawn data

    max_range: int
        maximal range of drawn data
    """
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    # Permute dims from NCHW/CHW to WHC
    tensor = np.transpose(tensor, (2, 1, 0))
    # Reshape data to have first dim with H*W size and second dim with 3 (R,G,B) channels
    tensor = np.reshape(tensor, (-1, 2))
    # Clip data to specified range (better visualization for outliers)
    tensor = np.clip(tensor, a_min=min_range, a_max=max_range)
    pcd.triangle_uvs = o3d.utility.Vector2dVector(tensor)
    o3d.visualization.draw_geometries([pcd])


def showPointCloud3D(
    tensor: TensorType = None, min_range: int = -1000, max_range: int = 1000
) -> None:
    r"""

    Show points cloud in 3D

    Attributes
    ----------
    tensor: TensorType (torch.tensor)
        CHW shaped pytorch tensor

    min_range: int
        minimal range of drawn data

    max_range: int
        maximal range of drawn data
    """
    assert tensor.dim() != 4, "provided tensor must be CHW shaped tensor, not NCHW!"
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    # Permute dims from NCHW/CHW to WHC
    tensor = np.transpose(tensor, (2, 1, 0))
    # Reshape data to have first dim with H*W size and second dim with 3 (R,G,B) channels
    tensor = np.reshape(tensor, (-1, 3))
    # Clip data to specified range (better visualization for outliers)
    tensor = np.clip(tensor, a_min=min_range, a_max=max_range)
    pcd.points = o3d.utility.Vector3dVector(tensor)
    o3d.visualization.draw_geometries([pcd])
