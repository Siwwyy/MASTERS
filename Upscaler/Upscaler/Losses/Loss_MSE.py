
import torch
import torch.nn                     as nn

from Losses.Loss_Base               import Loss_Base
from Config.Config                  import TensorType

class Loss_MSE(Loss_Base):
    def __init__(
        self,
        reduction: str ="mean"
    ):
        super().__init__("Loss_MSE")
        self.reduction = reduction

    def forward(self, pred: TensorType = None, target: TensorType = None) -> TensorType:
        assert pred is not None, "Input tensor pred can't be None!"
        assert target is not None, "Input tensor target can't be None!"
        
        if self.reduction == "none":
            return (pred - target).pow(2)
        elif self.reduction == "sum":
            return (pred - target).pow(2).sum()
        return (pred - target).pow(2).mean()


def test():
    pred                = torch.rand((3,20,20), dtype=torch.float32)
    target              = torch.rand((3,20,20), dtype=torch.float32)
    mseLoss             = nn.MSELoss()
    mseLossCustom       = Loss_MSE()

    assert torch.allclose(mseLoss(pred, target), mseLossCustom(pred, target))
    print(mseLoss(pred, target))
    print(mseLossCustom(pred, target))

    mseLoss             = nn.MSELoss(reduction="sum")
    mseLossCustom       = Loss_MSE(reduction="sum")
    assert torch.allclose(mseLoss(pred, target), mseLossCustom(pred, target))
    print(mseLoss(pred, target))
    print(mseLossCustom(pred, target))

    mseLoss             = nn.MSELoss(reduction="none")
    mseLossCustom       = Loss_MSE(reduction="none")
    assert torch.allclose(mseLoss(pred, target), mseLossCustom(pred, target))
    print(mseLoss(pred, target))
    print(mseLossCustom(pred, target))
    print(mseLossCustom(pred, target))