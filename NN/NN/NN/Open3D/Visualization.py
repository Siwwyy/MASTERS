from typing import List
from Config import TensorType, PathType

import torch
import open3d as o3d
import numpy as np

__all__ = ["showPointCloud3D"]


def showPointCloud3D(
    inputTensor: TensorType = None, min_range: int = -1000, max_range: int = 1000
) -> None:
    r"""

    Show points cloud in 3D

    Attributes
    ----------
    inputTensor: TensorType (torch.tensor)
        CHW shaped pytorch tensor

    min_range: int
        minimal range of drawn data

    max_range: int
        maximal range of drawn data
    """
    if torch.is_tensor(inputTensor):
        inputTensor = inputTensor.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    # Permute dims from NCHW/CHW to WHC
    inputTensor = np.transpose(inputTensor, (2, 1, 0))
    # Reshape data to have first dim with H*W size and second dim with 3 (R,G,B) channels
    inputTensor = np.reshape(inputTensor, (-1, 3))
    # Clip data to specified range (better visualization for outliers)
    inputTensor = np.clip(inputTensor, a_min=min_range, a_max=max_range)
    pcd.points = o3d.utility.Vector3dVector(inputTensor)
    o3d.visualization.draw_geometries([pcd])
