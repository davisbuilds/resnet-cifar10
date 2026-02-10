from __future__ import annotations

import pytest

from utils.config import load_experiment_config, validate_config


def test_load_baseline_config() -> None:
    config = load_experiment_config("experiments/baseline.yaml")
    assert config["experiment_name"] == "baseline"
    assert config["model"]["use_residual"] is True
    assert config["optimization"]["epochs"] == 150


def test_validate_config_rejects_missing_keys() -> None:
    broken = {"experiment_name": "x"}
    with pytest.raises(ValueError):
        validate_config(broken)  # type: ignore[arg-type]

