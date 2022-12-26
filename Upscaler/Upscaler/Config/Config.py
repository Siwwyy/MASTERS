from __future__ import annotations

import torch

from typing import Union, Annotated, Dict
from pathlib import Path

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

""" 
    Possible Path Type
"""
PathType = Union[str, Path]

""" 
    Possible Dict Type
"""
DictType = Union[dict, Dict]


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
        torch.cuda.is_available()
    ), "CUDA is not available, Check if you have driver/software updates or proper NVIDIA GPU"
    if torch.cuda.device_count() >= gpu_idx + 1:
        return torch.device(f"cuda:{gpu_idx}")
    return torch.device("cpu")


""" 
    Currently used device
"""
CurrentDevice: torch.device = try_gpu(gpu_idx=0)


from datetime import date
""" 
    Training ,inference ... results saving path
"""

def GetResultsPath(result_date:date=None) -> Path:
    if result_date is None: result_date = date.today()
    return_path = Path("E:/MASTERS/Upscaler/Results/{}".format(result_date)) 
    if not return_path.exists():
        return_path.mkdir()
    return return_path

ResultsPath = GetResultsPath()


def GetTrainingsPath(directory:Path=ResultsPath, stem:PathType="baseline") -> Path:
    return_path = Path(directory/"Trainings"/stem)
    if not return_path.exists():
        return_path.mkdir(parents=True)
    return return_path

TrainingsPath = GetTrainingsPath()


def GetInferenceResultsPath(directory:Path=ResultsPath, stem:PathType="baseline") -> Path:
    return_path = Path(directory/"InferenceResults"/stem)
    if not return_path.exists():
        return_path.mkdir(parents=True)
    return return_path

InferenceResultsPath = GetInferenceResultsPath()