import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from NN.Config.BaseTypes import PathType


def classic_npy_pipeline(cfg: DictConfig = None) -> bool:
    print(cfg)
    return True
