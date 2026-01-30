"""
Small utilities used across the repo-root `lvdm` package.

This repo previously relied on `utils_loc.utils` from the vendored scaling-noise tree.
To avoid that dependency, we provide the minimal subset here.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict


def get_obj_from_str(path: str) -> Any:
    """
    Import and return an object from a fully-qualified path, e.g.:
      "lvdm.models.ddpm3d.LatentDiffusion"
    """
    module, name = path.rsplit(".", 1)
    mod = importlib.import_module(module)
    return getattr(mod, name)


def instantiate_from_config(config: Dict[str, Any]) -> Any:
    """
    Instantiate an object from an OmegaConf/YAML-style config dict:
      {"target": "package.ClassName", "params": {...}}
    """
    if config is None:
        raise ValueError("instantiate_from_config got None config")
    if "target" not in config:
        raise KeyError("Expected key `target` in config to instantiate.")
    cls = get_obj_from_str(str(config["target"]))
    params = config.get("params", {}) or {}
    return cls(**params)


def count_params(model) -> int:
    """Return number of parameters (trainable + frozen)."""
    return sum(int(p.numel()) for p in model.parameters())

