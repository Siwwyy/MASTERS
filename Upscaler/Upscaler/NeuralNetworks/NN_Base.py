import torch
import torch.nn as nn
import torch.nn.functional as F

from abc                import ABCMeta, abstractmethod
from Config.Config      import TensorType, ShapeType
from typing             import Optional


class Model_Base(torch.nn.Module, metaclass=ABCMeta):
    def __init__(
        self, name: str = "Model_Base", input_shape: ShapeType = (1, 3, 1920, 1080)
    ):
        super().__init__()
        self.name = name
        self.input_shape = input_shape

    @abstractmethod
    def forward(self, x: TensorType = None) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.forward.__name__)
        )

    def __repr__(self):
        return self.name