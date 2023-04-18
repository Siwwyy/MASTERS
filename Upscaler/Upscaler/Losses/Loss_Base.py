import torch

from abc import ABCMeta, abstractmethod
from Config import TensorType

# from Config.config import TensorType


class Loss_Base(torch.nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        name: str = "Loss_Base",
    ):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, pred: TensorType = None, target: TensorType = None) -> TensorType:
        assert pred is not None, "Input tensor pred can't be None!"
        assert target is not None, "Input tensor target can't be None!"
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.forward.__name__)
        )

    def __repr__(self) -> str:
        return self.name
