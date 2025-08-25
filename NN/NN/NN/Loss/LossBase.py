import torch
import torch.nn as nn

import abc

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

    def __repr__(self) -> str:
        return self.name


if __name__ == "__main__":
    pass
    # abc = LossBase()
    # # print(abc.__static_attributes__)
    # # print()
    # print(abc(0, 1))
