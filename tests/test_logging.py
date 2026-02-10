from __future__ import annotations

from pathlib import Path

from utils.logging import ExperimentLogger


def test_experiment_logger_resume_appends_metrics(tmp_path: Path) -> None:
    logger = ExperimentLogger(tmp_path)
    logger.log_epoch(1, 1.0, 0.5, 1.2, 0.4, 0.1)

    resumed_logger = ExperimentLogger(tmp_path, resume=True)
    resumed_logger.log_epoch(2, 0.9, 0.6, 1.0, 0.5, 0.1)

    metrics_path = tmp_path / "metrics.csv"
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 3
    assert lines[0].startswith("epoch,train_loss,train_acc,val_loss,val_acc,lr")
    assert lines[1].startswith("1,1.0,0.5,1.2,0.4,0.1")
    assert lines[2].startswith("2,0.9,0.6,1.0,0.5,0.1")

