import copy
import os

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config["_config_path"] = os.path.abspath(path)
    return config
