import torch
import torch.nn as nn
import torchvision.transforms

from NN.Config import TensorType, PathType


__all__ = ["IdentityTransform"]

# Please keep this list sorted
assert __all__ == sorted(__all__)


class IdentityTransform(object):
    r"""
    Identity transform

    tens = IdentityTransform(tens1)
    tens == tens1

    Attributes
    ----------
    name : str
        Name of dataset
    ds_root_path : PathType (Union[str, Path])
        Root path to dataset
    ----------
    """

    def __init__(self):
        pass

    def __call__(self, tens: TensorType = None):
        """
        Parameters
        ----------
        tens : TensorType (Pytorch tensor)
            Input pytorch's tensor with arbitrary shape.
        Returns
        ---------
        Pytorch's tensor, identity of input!
        """
        assert tens is not None, "Input tens cant be none!"
        return tens

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
