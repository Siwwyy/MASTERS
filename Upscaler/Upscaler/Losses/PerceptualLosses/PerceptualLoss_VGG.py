import torch
import torch.nn as nn

from Losses.Loss_Base import Loss_Base
from Losses.Loss_MAE import Loss_MAE
from Config.Config import TensorType

from collections import namedtuple
from typing import List

from torchvision import transforms
from torchvision.models import vgg16, vgg19


#class VGG19(nn.Module):
#    def __init__(self):
#        super().__init__()

#        blocksSlice = [
#            slice(0, 4),
#            slice(5, 9),
#            slice(10, 18),
#            slice(19, 27),
#            slice(28, 36),
#        ]
#        vggblocksNamedTuple = namedtuple(
#            "vggBlocks", ["block1", "block2", "block3", "block4", "block5"]
#        )

#        # Get vgg model and turn on eval mode
#        vggModel = vgg19(pretrained=True).eval()
#        modelFeatures = vggModel.features

#        # Assign features (specified layers) from VGG19 to self
#        self.model = torch.nn.ModuleList(
#            [modelFeatures[slice] for slice in blocksSlice]
#        )

#        # Turn off requires_grad for parameters in VGG19 blocks
#        for param in self.model.parameters():
#            param.requires_grad = False

#    def forward(self, x: TensorType = None) -> List[TensorType]:
#        assert x is not None, "Input tensor X can't be None!"

#        # blockOutput = []
#        # for block in self.model:
#        #    x = block(x)
#        #    return x
#        # blockOutput.append(x)
#        # torch.cuda.empty_cache()

#        # return blockOutput
#        return self.model[0](x)
#        # return x


class PerceptualLoss_VGG(Loss_Base):


    models = {'vgg16': vgg16, 'vgg19': vgg19}

    def __init__(
        self,
        criterion: Loss_Base = None,
        normalizeInput: bool = False,
        reduction: str = "mean",
    ):
        super().__init__("PerceptualLoss_VGG")
        self.reduction = reduction
        self.criterion = criterion
        self.normalize = lambda x: x
        if normalizeInput:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0] #TODO, rethink weights -> how important specified layer is by pre-defined weights

        if self.criterion is None:
            self.criterion = Loss_MAE(reduction)

        # VGG19 network
        blocksSlice = [
            slice(0, 4),
            slice(5, 9),
            slice(10, 18),
            slice(19, 27),
            slice(28, 36),
        ]

        # Get vgg model and turn on eval mode
        vggModel = vgg19(pretrained=True).eval()
        modelFeatures = vggModel.features

        # Assign features (specified layers) from VGG19 to self
        #self.model = torch.nn.ModuleList(
        #    [modelFeatures[slice] for slice in blocksSlice]
        #)
        #self.model = torch.nn.ModuleList(modelFeatures[blocksSlice[0]])
        self.model = nn.Sequential(modelFeatures[blocksSlice[0]])

        # Turn off requires_grad for parameters in VGG19 blocks
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self, srPred: TensorType = None, hrTarget: TensorType = None
    ) -> TensorType:
        assert srPred is not None, "Input tensor pred can't be None!"
        assert hrTarget is not None, "Input tensor target can't be None!"

        #srPred = self.model(srPred)
        #hrTarget = self.model(hrTarget)

        srPred = self.model(self.normalize(srPred))
        hrTarget = self.model(self.normalize(hrTarget))
        return self.criterion(srPred, hrTarget)

        #loss = torch.tensor(
        #    [0.0], dtype=torch.float32, device=srPred.device, requires_grad=True
        #)  # loss will not work if reduction != mean
        #for feature in self.model:
        #    #if idx == 3: break #lets calculate vgg loss on first 3 layers (no available GPU Memory for deeper layers)
        #    srPred = feature(srPred)
        #    with torch.no_grad():
        #        hrTarget = feature(hrTarget.detach())
        #    loss = loss + self.criterion(srPred, hrTarget)
        #return loss





def test():
    pred = torch.rand((3, 20, 20), dtype=torch.float32)
    target = torch.rand((3, 20, 20), dtype=torch.float32)
    # maeLoss = nn.L1Loss()
    # maeLossCustom = Loss_MAE()

    # assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    # print(maeLoss(pred, target))
    # print(maeLossCustom(pred, target))

    # maeLoss = nn.L1Loss(reduction="sum")
    # maeLossCustom = Loss_MAE(reduction="sum")
    # assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    # print(maeLoss(pred, target))
    # print(maeLossCustom(pred, target))

    # maeLoss = nn.L1Loss(reduction="none")
    # maeLossCustom = Loss_MAE(reduction="none")
    # assert torch.allclose(maeLoss(pred, target), maeLossCustom(pred, target))
    # print(maeLoss(pred, target))
    # print(maeLossCustom(pred, target))
