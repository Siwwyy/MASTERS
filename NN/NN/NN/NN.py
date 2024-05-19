# Neural Network Engine

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from typing import TypeVar

from pathlib import Path
from Config.BaseTypes import PathType, _NNBaseClass
from Dataset import DatasetUE
from Loss import MSELoss

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T", bound=_NNBaseClass)


def createObjectfromConfig(cfg: DictConfig) -> T:
    return hydra.utils.instantiate(cfg)


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "Config/ConfigFiles",
    "config_name": "config.yaml",
}


@hydra.main(**_HYDRA_PARAMS)
def configMain(cfg: DictConfig) -> None:
    print(cfg)
    print(OmegaConf.to_yaml(cfg))
    convertedCfg = OmegaConf.to_yaml(cfg)

    # datasetType = cfg['dataset']

    # abc  = datasetType['_class_name_'](datasetType['datasetRootPath'], datasetType['csvPath'])
    dataset = createObjectfromConfig(cfg.dataset)
    loss = createObjectfromConfig(cfg.loss)
    # abc  = DatasetUE()


if __name__ == "__main__":
    configMain()
