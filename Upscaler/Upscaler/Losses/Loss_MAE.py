

import torch
import torch.nn                     as nn

from Losses.Loss_Base               import Loss_Base
from Config.Config                  import TensorType

# from Config.config import TensorType


class Loss_MAE(Loss_Base):
    def __init__(
        self,
        reduction="mean"
    ):
        super().__init__()
        self.name = "Loss_MAE"
        self.reduction = reduction

    def forward(self, x: TensorType = None, y_pred: TensorType = None) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"
        assert y_pred is not None, "Input tensor Y_pred can't be None!"
        
        if self.reduction == "none":
            return (x - y_pred).abs()
        elif self.reduction == "sum":
            return (x - y_pred).abs().sum()
        return (x - y_pred).abs().mean()


def test():
    pred                = torch.rand((3,20,20), dtype=torch.float32)
    target              = torch.rand((3,20,20), dtype=torch.float32)
    maeLoss             = nn.L1Loss()
    maeLossCustom       = Loss_MAE()

    assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    print(maeLoss(pred, target))
    print(maeLossCustom(pred, target))

    maeLoss             = nn.L1Loss(reduction="sum")
    maeLossCustom       = Loss_MAE(reduction="sum")
    assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    print(maeLoss(pred, target))
    print(maeLossCustom(pred, target))

    maeLoss             = nn.L1Loss(reduction="none")
    maeLossCustom       = Loss_MAE(reduction="none")
    assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    print(maeLoss(pred, target))
    print(maeLossCustom(pred, target))