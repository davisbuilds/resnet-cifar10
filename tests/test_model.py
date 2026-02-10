from __future__ import annotations

import torch

from models.blocks import BasicBlock
from models.resnet import resnet18_cifar


def test_resnet18_cifar_output_shape() -> None:
    model = resnet18_cifar(num_classes=10, use_residual=True)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    assert y.shape == (4, 10)


def test_residual_toggle_keeps_param_count_constant() -> None:
    model_res = resnet18_cifar(num_classes=10, use_residual=True)
    model_no_res = resnet18_cifar(num_classes=10, use_residual=False)
    params_res = sum(p.numel() for p in model_res.parameters())
    params_no_res = sum(p.numel() for p in model_no_res.parameters())
    assert params_res == params_no_res


def test_basic_block_residual_toggle_changes_output() -> None:
    block_res = BasicBlock(16, 16, stride=1, use_residual=True).eval()
    block_no_res = BasicBlock(16, 16, stride=1, use_residual=False).eval()
    block_no_res.load_state_dict(block_res.state_dict())

    with torch.no_grad():
        for block in (block_res, block_no_res):
            block.conv1.weight.zero_()
            block.conv2.weight.zero_()
            block.bn1.weight.fill_(1.0)
            block.bn1.bias.zero_()
            block.bn1.running_mean.zero_()
            block.bn1.running_var.fill_(1.0)
            block.bn2.weight.fill_(1.0)
            block.bn2.bias.zero_()
            block.bn2.running_mean.zero_()
            block.bn2.running_var.fill_(1.0)

    x = torch.randn(2, 16, 8, 8)
    y_res = block_res(x)
    y_no_res = block_no_res(x)

    assert torch.allclose(y_res, torch.relu(x), atol=1e-5)
    assert torch.allclose(y_no_res, torch.zeros_like(y_no_res), atol=1e-5)

