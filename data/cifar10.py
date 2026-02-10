from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def train_val_split_indices(train_size: int, val_size: int, split_seed: int) -> tuple[list[int], list[int]]:
    if val_size <= 0 or val_size >= train_size:
        raise ValueError(f"val_size must be in [1, {train_size - 1}], got {val_size}.")

    generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(train_size, generator=generator).tolist()
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    return train_indices, val_indices


def get_cifar10_train_val_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    val_size: int,
    split_seed: int,
) -> tuple[DataLoader, DataLoader]:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    train_dataset_full = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=True,
        transform=build_train_transform(),
    )
    val_dataset_full = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=True,
        transform=build_test_transform(),
    )

    train_indices, val_indices = train_val_split_indices(
        train_size=len(train_dataset_full),
        val_size=val_size,
        split_seed=split_seed,
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader


def get_cifar10_test_dataloader(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    test_dataset = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=True,
        transform=build_test_transform(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return test_loader
