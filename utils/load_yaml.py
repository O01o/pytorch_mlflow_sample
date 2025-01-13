import yaml

from entity.params import Config


def config_by_yaml(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)