from omegaconf import DictConfig, OmegaConf

import hydra
from hydra import compose, initialize


# @hydra.main(version_base=None, config_path="../ConfigFiles", config_name="config")
# def my_app(cfg):
#     print(OmegaConf.to_yaml(cfg))


@hydra.main(version_base=None, config_path="../ConfigFiles", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


# def LoadConfigFile(configPath:str=None, configFileName:str=None):
#     # assert configFileName is not None and configPath is not None, "Add logging here"
#     with initialize(version_base=None, config_path="../ConfigFiles", job_name="test"):
#         cfg = compose(config_name=configPath)
#         return OmegaConf.to_yaml(cfg)

# def LoadMainConfig(configName:str=None):
#     assert configName is not None, "Add logging here"
#     with initialize(version_base=None, config_path="../ConfigFiles", job_name="test"):
#         cfg = compose(config_name=configName)
#         print(type(cfg))
#         print(cfg)
#         loadedConfig = OmegaConf.to_yaml(cfg)
#         print(type(loadedConfig))


#         for key, value in cfg.items():
#             print(key, value)
#             cfg[key].__add__(LoadConfigFile(value))


#     print(OmegaConf.to_yaml(cfg))


#         # LoadConfig()


if __name__ == "__main__":
    print(__name__)
    my_app()
    # LoadMainConfig("config")
