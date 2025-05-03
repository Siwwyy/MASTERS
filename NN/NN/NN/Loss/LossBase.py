import torch
import torch.nn as nn

from NN.Config.BaseTypes import _NNabstractMethod, TensorType, PathType, _NNBaseClass

__all__ = ["LossBase"]


class LossBase(_NNBaseClass):
    def __init__(
        self,
        name: str = "LossBase",
    ):
        super().__init__()
        self.name = name

    @_NNabstractMethod
    def forward(self, pred: TensorType = None, target: TensorType = None) -> TensorType:
        assert pred is not None, "Input tensor pred can't be None!"
        assert target is not None, "Input tensor target can't be None!"
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.forward.__name__)
        )

    def __repr__(self) -> str:
        return self.name
