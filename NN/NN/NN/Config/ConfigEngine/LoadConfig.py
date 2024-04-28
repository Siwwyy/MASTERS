import hydra

from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize


@hydra.main(version_base=None, config_path="../ConfigFiles", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
