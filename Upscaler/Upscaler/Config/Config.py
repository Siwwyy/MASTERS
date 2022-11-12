from __future__ import annotations

import torch

from typing import Union, Annotated

# TensorType = Annotated[torch.tensor, "Possible Tensor type"]
# ShapeType = Annotated[Union[tuple, torch.Size], "Possible Shape types of the
# tensor"]

""" 
    Possible Tensor type
"""
TensorType = torch.tensor


""" 
    Possible Shape types of the tensor
"""
ShapeType = Union[tuple, torch.Size]


def try_gpu(gpu_idx: int = 0) -> torch.device:
    """Returns a device (GPU) on specified gpu_idx,
        If GPU does not exist, then it returns CPU

    Parameters
    ----------
    gpu_idx : int
        GPU Device idx, number of specified GPU
    Returns
    -------
        function returns GPU on specified index if exists, if not, CPU."""
    assert (
        not torch.cuda.is_available()
    ), "CUDA is not available, Check if you have driver/software updates or proper NVIDIA GPU"
    if torch.cuda.device_count() >= gpu_idx + 1:
        return torch.device(f"cuda:{gpu_idx}")
    return torch.device("cpu")


""" 
    Currently used device
"""
CurrentDevice: torch.device = try_gpu(gpu_idx=0)
