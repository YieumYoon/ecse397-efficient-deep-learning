"""ResNet-18 model helpers for CIFAR-10 experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def create_resnet18(
    num_classes: int = 10,
    pretrained: bool = False,
    in_channels: int = 3,
) -> nn.Module:
    """Instantiate a torchvision ResNet-18 tailored for CIFAR-10."""

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if in_channels != 3:
        old_conv = model.conv1
        bias = old_conv.bias is not None
        model.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=bias,
        )
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
        if bias:
            nn.init.zeros_(model.conv1.bias)
        if pretrained:
            with torch.no_grad():
                if in_channels == 1:
                    model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                elif in_channels == 3:
                    model.conv1.weight.copy_(old_conv.weight)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


__all__ = ["create_resnet18"]
