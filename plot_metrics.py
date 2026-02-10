from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics from one or more run directories.")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more results run directories containing metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write plots. Defaults to <run_dir>/plots for one run.",
    )
    return parser.parse_args()


def read_metrics(path: Path) -> dict[str, list[float]]:
    fields = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr")
    data: dict[str, list[float]] = {field: [] for field in fields}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            data["epoch"].append(float(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["train_acc"].append(float(row["train_acc"]))
            data["val_loss"].append(float(row["val_loss"]))
            data["val_acc"].append(float(row["val_acc"]))
            data["lr"].append(float(row["lr"]))
    return data


def resolve_output_dir(run_dirs: list[Path], output_dir: str | None) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    if len(run_dirs) == 1:
        path = run_dirs[0] / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("results") / f"plots_compare_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss(run_metrics: list[tuple[str, dict[str, list[float]]]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name, metrics in run_metrics:
        sns.lineplot(x=metrics["epoch"], y=metrics["train_loss"], ax=ax, label=f"{run_name} train")
        sns.lineplot(x=metrics["epoch"], y=metrics["val_loss"], ax=ax, label=f"{run_name} val")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=180)
    plt.close(fig)


def plot_accuracy(run_metrics: list[tuple[str, dict[str, list[float]]]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name, metrics in run_metrics:
        sns.lineplot(x=metrics["epoch"], y=metrics["train_acc"], ax=ax, label=f"{run_name} train")
        sns.lineplot(x=metrics["epoch"], y=metrics["val_acc"], ax=ax, label=f"{run_name} val")
    ax.set_title("Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dirs = [Path(run_dir) for run_dir in args.run_dirs]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise ValueError(f"Run directory does not exist: {run_dir}")
        if not (run_dir / "metrics.csv").exists():
            raise ValueError(f"No metrics.csv found in run directory: {run_dir}")

    sns.set_theme(style="whitegrid")
    output_dir = resolve_output_dir(run_dirs, args.output_dir)

    run_metrics: list[tuple[str, dict[str, list[float]]]] = []
    for run_dir in run_dirs:
        metrics = read_metrics(run_dir / "metrics.csv")
        run_metrics.append((run_dir.name, metrics))

    plot_loss(run_metrics, output_dir)
    plot_accuracy(run_metrics, output_dir)
    print(f"plots_dir={output_dir}")


if __name__ == "__main__":
    main()

