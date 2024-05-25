from __future__ import annotations
from re import L
from typing import Union, Annotated, Mapping, Any
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
    Currently used objects in NN library
"""
_DECLARED_OBJECTS_: list[_NNMetaClass] = []

""" 
    Meta class of every class in NN library
"""


class _NNMetaClass(type):
    def __new__(
        metacls,
        name: str,
        bases: tuple[_NNMetaClass, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ):
        """
        Creates new instance

        metacls: _NNMetaClass
            The metaclass itself
        name: str
            name of the class
        bases: tuple[_NNMetaClass, ...]
            base classes of the class
        namespace: dict[str, Any]
            attributes of class e.g., __doc__, functions dict, variables dict and so on
        kwargs: Any
            additional keyword arguments (for now, not used)
        """
        print(
            "  Meta.__new__(mcs=%s, name=%r, bases=%s, attrs=[%s], **%s)"
            % (metacls, name, bases, ", ".join(namespace), kwargs)
        )
        return super().__new__(metacls, name, bases, namespace)

    @classmethod
    def __prepare__(
        metacls, name: str, bases: tuple[_NNMetaClass, ...], **kwargs: Any
    ) -> Mapping[str, object]:
        """
        Prepares class

        metacls: _NNMetaClass
            The metaclass itself
        name: str
            name of the class
        bases: tuple[_NNMetaClass, ...]
            base classes of the class
        kwargs: Any
            additional keyword arguments (for now, not used)
        """
        print(
            "  Meta.__prepare__(mcs=%s, name=%r, bases=%s, **%s)"
            % (metacls, name, bases, kwargs)
        )
        return super().__prepare__(metacls, name, bases, **kwargs)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        print("  Meta.__call__(cls=%s, *%s, **%s)" % (cls, args, kwargs))
        _DECLARED_OBJECTS_.append(super().__call__(*args, **kwargs))
        return _DECLARED_OBJECTS_[-1]

    def __del__(cls):
        # print("DDDD")
        # for obj in _DECLARED_OBJECTS_:
        #     if obj is cls:
        #         _DECLARED_OBJECTS_.remove(obj)

        # print(_DECLARED_OBJECTS_[0])
        pass


""" 
    Base class of every class in NN library
"""


class _NNBaseClass(torch.nn.Module, metaclass=_NNMetaClass):
    ...
