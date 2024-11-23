import math
from typing import Union, Annotated, Dict
from pathlib import Path

import torch

__all__ = ["try_gpu"]

# Please keep this list sorted
assert __all__ == sorted(__all__)


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
    # Fallback to CPU if cuda is not available
    if not torch.cuda.is_available():
        print(
            "CUDA is not available, Check if you have driver/software updates or proper NVIDIA GPU installed"
        )
        return torch.device("cpu")

    if torch.cuda.device_count() >= gpu_idx + 1:
        return torch.device(f"cuda:{gpu_idx}")
    return torch.device("cpu")


def DIV_UP(nominator: int, denominator: int) -> int | float:
    """Returns rounded up value (useful for getting blocks/threads amount)
    Math formula: floor((nominator + denominator - 1) / denominator)
    Parameters
    ----------
    nominator : int
        nominator in math formula

    denominator: int
        denominator in math formula
    Returns
    -------
        function returns rounded up + floor to closest divisior"""
    return math.floor((nominator + denominator - 1) / denominator)
