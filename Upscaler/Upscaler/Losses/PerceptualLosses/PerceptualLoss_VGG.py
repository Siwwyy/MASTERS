import torch
import torch.nn as nn

from Losses.Loss_Base import Loss_Base
from Config.Config import TensorType
from torchvision.models import vgg19
from collections import namedtuple

class VGG19(nn.Module):
    def __init__(self):
        super().__init__()

        blocksSlice = [ slice(0, 4), slice(5, 9), slice(10, 18), slice(19, 27), slice(28, 36)  ]
        vggblocksNamedTuple = namedtuple("vggBlocks", ["block1", "block2", "block3", "block4", "block5"])

        self.model = vgg19(pretrained=True)
        self.modelFeatures = self.model.features
        # Get VGG19 features and turn eval mode (no gradients needed)
        self.vggBlocks = vggblocksNamedTuple(*[ self.modelFeatures[slice].eval() for slice in blocksSlice])


class PerceptualLoss_VGG(Loss_Base):
    def __init__(self, reduction: str = "mean"):
        super().__init__("PerceptualLoss_VGG")
        self.reduction = reduction

    def forward(self, srPred: TensorType = None, hrTarget: TensorType = None) -> TensorType:
        assert srPred is not None, "Input tensor pred can't be None!"
        assert hrTarget is not None, "Input tensor target can't be None!"

        


def test():
    pred = torch.rand((3, 20, 20), dtype=torch.float32)
    target = torch.rand((3, 20, 20), dtype=torch.float32)
    #maeLoss = nn.L1Loss()
    #maeLossCustom = Loss_MAE()

    #assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    #print(maeLoss(pred, target))
    #print(maeLossCustom(pred, target))

    #maeLoss = nn.L1Loss(reduction="sum")
    #maeLossCustom = Loss_MAE(reduction="sum")
    #assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    #print(maeLoss(pred, target))
    #print(maeLossCustom(pred, target))

    #maeLoss = nn.L1Loss(reduction="none")
    #maeLossCustom = Loss_MAE(reduction="none")
    #assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    #print(maeLoss(pred, target))
    #print(maeLossCustom(pred, target))

