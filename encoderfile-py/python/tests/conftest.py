import os
from pathlib import Path

import yaml
import json


def asset_path(filename: str) -> str:
    """
    Returns the absolute path to an asset file in the assets directory,
    regardless of the current working directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "assets", filename)


def load_yaml_asset(filename):
    """
    Loads a yaml asset file from the assets directory.
    """
    path = asset_path(filename)
    if filename.endswith((".yml", ".yaml")):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Only yaml files are supported for this fixture.")


def load_json(path: Path):
    """
    Loads a json asset file from the assets directory.
    """
    if path.suffix != ".json":
        raise ValueError("Only json files are supported for this fixture.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
