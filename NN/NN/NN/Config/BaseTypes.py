from __future__ import annotations
from typing import Union, Annotated
from pathlib import Path
from .ConfigUtils import try_gpu

import torch


__all__ = ["TensorType", "ShapeType", "PathType", "DictType", "CurrentDevice"]

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
DictType = dict

""" 
    Currently used device
"""
CurrentDevice: torch.device = try_gpu(gpu_idx=0)
