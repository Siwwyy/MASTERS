from .Config import PathType, DictType
from omegaconf import OmegaConf


def read_cfg(yaml_path: PathType = None) -> DictType:
    assert yaml_path is not None, "read_cfg, yaml_path param is None!"
    return OmegaConf.load(yaml_path)
