from Config import TensorType, PathType

import torch
import open3d as o3d
import numpy as np

__all__ = ["showPointCloud"]


def showPointCloud(
    data: TensorType = None, min_range: int = -1000, max_range: int = 1000
) -> None:
    r"""
    showPointCloud
    """
    if isinstance(data, TensorType):
        data = torch.from_numpy(data)

    pcd = o3d.geometry.PointCloud()
    # Permute dims from NCHW/CHW to WHC
    data = np.transpose(data, (2, 1, 0))
    # Reshape data to have first dim with H*W size and second dim with 3 (R,G,B) channels
    data = np.reshape(data, (-1, 3))
    # Clip data to specified range (better visualization for outliers)
    data = np.clip(data, a_min=min_range, a_max=max_range)
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])
