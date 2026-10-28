"""
Loads config.yaml into a simple nested object so callers can do
config.size_filter.max_aspect instead of config["size_filter"]["max_aspect"].

Requires PyYAML:  pip install pyyaml
"""

import yaml


class ConfigNode:
    """Turns a nested dict into nested attribute access, recursively."""

    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = ConfigNode(value)
            setattr(self, key, value)

    def __repr__(self):
        return f"ConfigNode({self.__dict__})"


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigNode(raw)


if __name__ == "__main__":
    cfg = load_config()
    print("paths.mask:", cfg.paths.mask)
    print("size_filter.max_aspect:", cfg.size_filter.max_aspect)
    print("yolo.tile_size:", cfg.yolo.tile_size)