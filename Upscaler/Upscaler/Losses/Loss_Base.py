import torch

from abc import ABCMeta, abstractmethod

# from Config.config import TensorType


class Loss_Base(torch.nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        name: str = "Loss_Base",
    ):
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x: TensorType = None, y_pred: TensorType = None) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"
        assert y_pred is not None, "Input tensor Y_pred can't be None!"
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.forward.__name__)
        )
