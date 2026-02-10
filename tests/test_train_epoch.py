from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from models.resnet import resnet18_cifar
from train import run_epoch


def test_train_epoch_runs_and_returns_metrics() -> None:
    torch.manual_seed(0)
    model = resnet18_cifar(num_classes=10, use_residual=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)

    train_loss, train_acc = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
        optimizer=optimizer,
        limit_batches=2,
    )

    assert train_loss > 0.0
    assert 0.0 <= train_acc <= 1.0

