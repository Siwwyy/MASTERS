from omegaconf import DictConfig, OmegaConf

from NN.Config.ConfigUtils.Utils import CreateObjectfromConfig
from NN.Dataset import DatasetBase

__all__ = ["create_dataset"]


def create_dataset(cfg: DictConfig = None) -> DatasetBase:
    ds = CreateObjectfromConfig(cfg)
    return ds
    # print(cfg.scenes)
