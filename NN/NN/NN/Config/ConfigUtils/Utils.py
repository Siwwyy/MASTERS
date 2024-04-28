from typing import Union, Annotated, Dict
from pathlib import Path

import torch

# Please keep this list sorted
__all__ = ["try_gpu"]


def try_gpu(gpu_idx: int = 0) -> torch.device:
    """Returns a device (GPU) on specified gpu_idx,
        If GPU does not exist, then it returns CPU

    Parameters
    ----------
    gpu_idx : int
        GPU Device idx, number of specified GPU, gpu_idx = 0 by default
    Returns
    -------
        function returns GPU on specified index if exists, if not, CPU."""
    assert (
        torch.cuda.is_available()
    ), "CUDA is not available, Check if you have driver/software updates or proper NVIDIA GPU"
    if torch.cuda.device_count() >= gpu_idx + 1:
        return torch.device(f"cuda:{gpu_idx}")
    return torch.device("cpu")
