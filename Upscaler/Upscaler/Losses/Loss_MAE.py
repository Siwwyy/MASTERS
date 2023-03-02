import torch
import torch.nn as nn

from Losses.Loss_Base import Loss_Base
from Config.Config import TensorType


class Loss_MAE(Loss_Base):
    def __init__(self, reduction: str = "mean"):
        super().__init__("Loss_MAE")
        self.reduction = reduction

    def forward(self, pred: TensorType = None, target: TensorType = None) -> TensorType:
        assert pred is not None, "Input tensor pred can't be None!"
        assert target is not None, "Input tensor target can't be None!"

        if self.reduction == "none":
            return (pred - target).abs()
        elif self.reduction == "sum":
            return (pred - target).abs().sum()
        return (pred - target).abs().mean()


def test():
    pred = torch.rand((3, 20, 20), dtype=torch.float32)
    target = torch.rand((3, 20, 20), dtype=torch.float32)
    maeLoss = nn.L1Loss()
    maeLossCustom = Loss_MAE()

    assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    print(maeLoss(pred, target))
    print(maeLossCustom(pred, target))

    maeLoss = nn.L1Loss(reduction="sum")
    maeLossCustom = Loss_MAE(reduction="sum")
    assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    print(maeLoss(pred, target))
    print(maeLossCustom(pred, target))

    maeLoss = nn.L1Loss(reduction="none")
    maeLossCustom = Loss_MAE(reduction="none")
    assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    print(maeLoss(pred, target))
    print(maeLossCustom(pred, target))
