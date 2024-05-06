import hydra

from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize


# from Dataset import DatasetUE

__all__ = ["configMain"]


@hydra.main(version_base=None, config_path="../ConfigFiles", config_name="config")
def configMain(cfg: DictConfig) -> None:
    print(cfg)
    print(OmegaConf.to_yaml(cfg))

    # abc  = DatasetUE()
    # abc  = DatasetUE()


if __name__ == "__main__":
    configMain()
