from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from data.cifar10 import get_cifar10_test_dataloader
from models.resnet import resnet18_cifar
from train import run_epoch, select_device


def per_class_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[int, float]:
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for class_idx in range(num_classes):
                class_mask = labels == class_idx
                total[class_idx] += class_mask.sum().cpu()
                correct[class_idx] += ((preds == labels) & class_mask).sum().cpu()

    result: dict[int, float] = {}
    for class_idx in range(num_classes):
        denom = int(total[class_idx].item())
        if denom == 0:
            result[class_idx] = 0.0
        else:
            result[class_idx] = float(correct[class_idx].item()) / denom
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.")
    parser.add_argument("--data-dir", default="data/cache", help="Dataset root directory.")
    parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--per-class", action="store_true", help="Report per-class accuracy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config")
    if config is None:
        raise ValueError("Checkpoint is missing 'config' payload.")

    num_classes = int(config["data"]["num_classes"])
    use_residual = bool(config["model"]["use_residual"])

    model = resnet18_cifar(num_classes=num_classes, use_residual=use_residual).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loader = get_cifar10_test_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    if args.per_class:
        class_acc = per_class_accuracy(model, test_loader, device, num_classes)
        for class_idx, acc in class_acc.items():
            print(f"class_{class_idx}_acc={acc:.4f}")


if __name__ == "__main__":
    main()
