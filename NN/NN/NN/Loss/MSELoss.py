import torch
import torch.nn as nn

from NN.Loss.LossBase import LossBase
from NN.Config import TensorType

__all__ = ["MSELoss"]


class MSELoss(LossBase):
    def __init__(self, reduction: str = "mean"):
        super().__init__("MSELoss")
        self.reduction = reduction

    def forward(self, pred: TensorType = None, target: TensorType = None) -> TensorType:
        assert pred is not None, "Input tensor pred can't be None!"
        assert target is not None, "Input tensor target can't be None!"

        if self.reduction == "none":
            return (pred - target).pow(2)
        elif self.reduction == "sum":
            return (pred - target).pow(2).sum()
        return (pred - target).pow(2).mean()


if __name__ == "__main__":
    # pass
    abc = MSELoss()
    # print(abc(0,1))
