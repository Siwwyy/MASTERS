import hydra
import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
from functools import partial
from typing import TypeVar, Union
from NN.Config.ConfigUtils.Utils import CreateObjectfromConfig


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../../Config/ConfigFiles/generate_npy",
    "config_name": "config.yaml",
}


@hydra.main(**_HYDRA_PARAMS)
def ConfigMain(cfg: DictConfig) -> None:
    """
    Main function for npy generation
    """
    npy_generation_function = CreateObjectfromConfig(cfg.npy_generator.func)
    npy_generation_function(cfg)


if __name__ == "__main__":
    ConfigMain()
