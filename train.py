from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from data.cifar10 import get_cifar10_train_val_dataloaders
from models.resnet import resnet18_cifar
from utils.config import load_experiment_config
from utils.logging import ExperimentLogger, current_git_commit, timestamped_output_dir
from utils.metrics import AverageMeter, accuracy_top1
from utils.seed import seed_everything


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, config: dict) -> SGD:
    optim_name = config["optimization"]["optimizer"].lower()
    if optim_name != "sgd":
        raise ValueError(f"Unsupported optimizer '{optim_name}'. Only SGD is implemented.")
    return SGD(
        model.parameters(),
        lr=float(config["optimization"]["lr"]),
        momentum=float(config["optimization"]["momentum"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    limit_batches: int | None = None,
) -> tuple[float, float]:
    train_mode = optimizer is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        images = images.to(device, non_blocking=False)
        targets = targets.to(device, non_blocking=False)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, targets)

        batch_acc = accuracy_top1(logits, targets)
        batch_size = targets.size(0)
        loss_meter.update(float(loss.item()), n=batch_size)
        acc_meter.update(batch_acc, n=batch_size)

    return loss_meter.avg, acc_meter.avg


def save_checkpoint(
    output_dir: Path,
    name: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: MultiStepLR,
    best_val_acc: float,
    config: dict,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "config": config,
    }
    torch.save(payload, output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 ResNet-18 experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument("--data-dir", default="data/cache", help="Dataset root directory.")
    parser.add_argument("--output-root", default="results", help="Directory for run outputs.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit run output directory. If omitted, a timestamped directory is created.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a checkpoint (.pt) to resume training from.",
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Limit train batches per epoch for smoke tests.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Limit validation batches per epoch for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    seed_everything(int(config["seed"]))
    device = select_device()

    resume_payload: dict | None = None
    resume_path: Path | None = None
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise ValueError(f"Checkpoint to resume from does not exist: {resume_path}")
        resume_payload = torch.load(resume_path, map_location=device)
        checkpoint_config = resume_payload.get("config")
        if checkpoint_config is not None and checkpoint_config != config:
            raise ValueError("Resume checkpoint config does not match provided --config file.")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    elif resume_path is not None:
        output_dir = resume_path.parent
    else:
        output_dir = timestamped_output_dir(args.output_root, config["experiment_name"])

    if resume_payload is None and (output_dir / "metrics.csv").exists():
        raise ValueError(
            f"Refusing to overwrite existing metrics at {output_dir / 'metrics.csv'}. "
            "Use a new output directory or pass --resume-from."
        )

    logger = ExperimentLogger(output_dir, resume=resume_payload is not None)
    logger.log_metadata(
        {
            "config_path": args.config,
            "resolved_device": str(device),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "git_commit": current_git_commit(),
            "resume_from": str(resume_path) if resume_path is not None else None,
            "config": config,
        }
    )

    val_size = int(config["data"].get("val_size", 5000))
    split_seed = int(config["data"].get("split_seed", config["seed"]))
    train_loader, val_loader = get_cifar10_train_val_dataloaders(
        data_dir=args.data_dir,
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        val_size=val_size,
        split_seed=split_seed,
    )

    model = resnet18_cifar(
        num_classes=int(config["data"]["num_classes"]),
        use_residual=bool(config["model"]["use_residual"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = MultiStepLR(
        optimizer,
        milestones=list(config["optimization"]["lr_schedule"]["milestones"]),
        gamma=float(config["optimization"]["lr_schedule"]["gamma"]),
    )

    start_epoch = 1
    best_val_acc = 0.0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best_val_acc = float(resume_payload.get("best_val_acc", 0.0))
        print(
            f"resuming_from={resume_path} start_epoch={start_epoch} best_val_acc={best_val_acc:.4f}",
            flush=True,
        )

    planned_epochs = int(config["optimization"]["epochs"])
    if args.max_epochs is not None:
        planned_epochs = min(planned_epochs, args.max_epochs)
    if start_epoch > planned_epochs:
        print(
            f"resume checkpoint is already at epoch {start_epoch - 1}, "
            f"which is >= planned_epochs={planned_epochs}.",
            flush=True,
        )
        print(f"results_dir={output_dir}", flush=True)
        return

    for epoch in range(start_epoch, planned_epochs + 1):
        lr = float(optimizer.param_groups[0]["lr"])
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            limit_batches=args.limit_train_batches,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            limit_batches=args.limit_val_batches,
        )
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                output_dir=output_dir,
                name="best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_acc=best_val_acc,
                config=config,
            )

        scheduler.step()

        print(
            f"epoch={epoch} lr={lr:.5f} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

    save_checkpoint(
        output_dir=output_dir,
        name="final.pt",
        epoch=planned_epochs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        best_val_acc=best_val_acc,
        config=config,
    )
    print(f"results_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
