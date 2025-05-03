from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from NN.Config import TensorType, ShapeType, _NNBaseClass

"""
Dataclass which encapsulates additional input informations to the model
"""


@dataclass
class ModelInputs:
    inputShape: ShapeType = (1, 3, 1920, 1080)  # NCHW


class ModelBase(_NNBaseClass):
    """
    ModelBase
    """

    def __init__(self, name: str = "Model_Base", modelInputs: ModelInputs = None):
        super().__init__()
        self.name = name
        self.inputShape = modelInputs.inputShape

    def forward(self, x: TensorType = None) -> TensorType:
        assert x is not None, "Input tensor X can't be None!"
        raise NotImplementedError(
            "Child class have to implement {} method".format(self.forward.__name__)
        )

    def __repr__(self):
        return self.name
