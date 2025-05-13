from __future__ import annotations
from typing import Callable, Union, Annotated, Mapping, Any
from pathlib import Path
from NN.Config.ConfigUtils import try_gpu

import torch

__all__ = [
    "TensorType",
    "ShapeType",
    "PathType",
    "DictType",
    "CurrentDevice",
    "_NNBaseClass",
]

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

################################################
# Metaclass, base class, decorators etc. section
################################################

""" 
    Currently used objects in NN library
"""
_DECLARED_CLASSES_: list[_NNMetaClass] = []

""" 
    Currently used objects in NN library
"""
_DECLARED_OBJECTS_: list[_NNBaseClass] = []

""" 
    Meta class of every class in NN library
"""

_VERBOSE_METACLASS: bool = True  # TODO: make this configurable


class _NNMetaClass(type):

    _ABSTRACT_METHODS_: dict[str, str] = {}

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
        if _VERBOSE_METACLASS:
            print(
                "  Meta.__new__(mcs=%s, name=%r, bases=%s, attrs=[%s], **%s)"
                % (metacls, name, bases, ", ".join(namespace), kwargs)
            )
        for key, value in namespace.items():
            ...
            # print(key, value)
            # print(getattr(value, '_isAbstractMethod_'))
            # if hasattr(value, '_isAbstractMethod_'):
            #     print("ABSTRACT!")
            #     raise RuntimeError("Can't instantiate class with abstract method")
            # if isinstance(value, types.FunctionType):
            #     print(dir(value))
            #     print(getattr(value, '_isAbstractMethod_'))
            #     if hasattr
            #     ...

        # _DECLARED_CLASSES_.append(newClass)
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
        if _VERBOSE_METACLASS:
            print(
                "  Meta.__prepare__(mcs=%s, name=%r, bases=%s, **%s)"
                % (metacls, name, bases, kwargs)
            )
        return super().__prepare__(metacls, name, bases, **kwargs)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if _VERBOSE_METACLASS:
            print("  Meta.__call__(cls=%s, *%s, **%s)" % (cls, args, kwargs))
        # TODO: Make ability to create object from str
        # _DECLARED_OBJECTS_.append(newObject)
        for _, value in cls.__dict__.items():
            if hasattr(value, "_isAbstractMethod_"):
                raise RuntimeError("Can't instantiate class with an abstract method")

        return super().__call__(*args, **kwargs)

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
    def __init__(self):
        super().__init__()


"""
    Decorators of NN API
"""


def _NNabstractMethod(func: Callable) -> Callable:
    setattr(func, "_isAbstractMethod_", True)
    return func
