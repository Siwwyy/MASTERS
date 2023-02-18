
import torch
import torch.nn                     as nn   

from Losses.Loss_Base               import Loss_Base
from Config.Config                  import TensorType
from typing                         import Union, Optional, List

class Loss_Combined(Loss_Base):
    def __init__(
        self,
        criterions: Optional[Union[Loss_Base, torch.nn.modules.loss._Loss]]=None,
        criterionContribution: Optional[Union[List[float], List[torch.tensor]]]=None
    ):
        super().__init__("Loss_Combined")
        self.criterions = criterions
        self.criterionContribution = criterionContribution

        if self.criterions is None:
            self.criterions = [ nn.MSELoss() ]
            self.criterionContribution = [ 1.0 ]

        assert len(self.criterions) == len(self.criterionContribution), "Amount of criterions must match criterionContribution amount"

    def forward(self, pred: TensorType = None, target: TensorType = None) -> TensorType:
        assert pred is not None, "Input tensor pred can't be None!"
        assert target is not None, "Input tensor target can't be None!"
        
        finalLoss = torch.tensor([0.0], dtype=torch.float32, device=pred.device, requires_grad=True)
        for criterion, criterionContribution in zip(self.criterions, self.criterionContribution):
            finalLoss = finalLoss + criterionContribution * criterion(pred, target)

        return finalLoss
            


def test():
    pred                = torch.rand((3,20,20), dtype=torch.float32)
    target              = torch.rand((3,20,20), dtype=torch.float32)
    loss                = Loss_Combined()


    print(loss(pred, target))
    print(loss(pred, target))