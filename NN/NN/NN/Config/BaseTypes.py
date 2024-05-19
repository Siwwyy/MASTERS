from __future__ import annotations
from typing import Union, Annotated
from pathlib import Path
from .ConfigUtils import try_gpu

import torch


__all__ = ["TensorType", "ShapeType", "PathType", "DictType", "CurrentDevice"]

# TensorType = Annotated[torch.tensor, "Possible Tensor type"]
# ShapeType = Annotated[Union[tuple, torch.Size], "Possible Shape types of the
# tensor"]

# After Python 3.12, "TensorType = torch.tensor" can be "type TensorType = torch.tensor" etc.
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

""" 
    Meta class of every class in NN library
"""


class _NNMetaClass(type):
    ...


""" 
    Base class of every class in NN library
"""


class _NNBaseClass(torch.nn.Module, metaclass=_NNMetaClass):
    ...
