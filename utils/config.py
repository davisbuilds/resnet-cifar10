from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_top = ("experiment_name", "seed", "data", "model", "optimization")
    for key in required_top:
        if key not in config:
            raise ValueError(f"Missing required top-level key: {key}")

    data_required = ("batch_size", "num_workers", "num_classes", "val_size", "split_seed")
    for key in data_required:
        if key not in config["data"]:
            raise ValueError(f"Missing data key: {key}")

    if "use_residual" not in config["model"]:
        raise ValueError("Missing model key: use_residual")

    optim_required = ("optimizer", "lr", "momentum", "weight_decay", "epochs", "lr_schedule")
    for key in optim_required:
        if key not in config["optimization"]:
            raise ValueError(f"Missing optimization key: {key}")

    schedule_required = ("milestones", "gamma")
    for key in schedule_required:
        if key not in config["optimization"]["lr_schedule"]:
            raise ValueError(f"Missing lr_schedule key: {key}")
