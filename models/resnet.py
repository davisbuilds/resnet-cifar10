from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks import BasicBlock


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        num_classes: int = 10,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, use_residual=use_residual)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_residual=use_residual)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_residual=use_residual)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_residual=use_residual)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int,
        use_residual: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                use_residual=use_residual,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=1,
                    use_residual=use_residual,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_cifar(num_classes: int = 10, use_residual: bool = True) -> ResNetCIFAR:
    return ResNetCIFAR(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        use_residual=use_residual,
    )

