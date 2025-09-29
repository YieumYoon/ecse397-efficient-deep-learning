"""ResNet-18 model helpers for CIFAR-10 experiments.

Background:
    ResNet-18 is a classic Convolutional Neural Network (CNN).  The key
    idea behind a ResNet is the *residual block*, which adds a skip
    connection so gradients can flow through deep networks without
    vanishing.  Each block learns the residual (the difference) between
    the input and the desired output, allowing the model to stack many
    layers while still being trainable.

    The original ResNet-18 was designed for ImageNet (1000 classes,
    224x224 images).  CIFAR-10 is smaller (10 classes, 32x32 images), so
    we adapt the final classification head and optionally reuse
    ImageNet pretraining to boost performance.

Purpose in Lab 1:
    - Establish a high-accuracy CNN baseline before pruning.
    - Provide weights and layer definitions so the pruning module can
      iterate over convolutional filters.

Learning tip:
    Try printing ``model`` after calling ``create_resnet18`` to inspect
    the architecture.  Understanding where the convolutional layers and
    batch norms live will help later when you analyse pruning masks.
"""

# from __future__ import annotations  # Commented for Python 3.6 compatibility

import torch  # Provides tensor operations and autograd.
from torch import nn  # Neural network modules and layers.
from torchvision import models  # Access to torchvision's model zoo.


def create_resnet18(
    num_classes: int = 10,
    pretrained: bool = False,
    in_channels: int = 3,
) -> nn.Module:
    """Instantiate a torchvision ResNet-18 tailored for CIFAR-10.

    Args:
        num_classes: Number of output logits.  CIFAR-10 uses 10.
        pretrained: Load ImageNet weights when ``True``.  Fine-tuning
            from pre-trained weights is the fastest path to the
            accuracy thresholds specified in the lab document.
        in_channels: Input channel count.  CIFAR-10 images are RGB, but
            this parameter makes the function flexible should you want
            to experiment with different modalities later.

    Returns:
        A ``torch.nn.Module`` ready for the training loop in
        ``train/train_loop.py``.
    """

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None  # Optional ImageNet weights.
    model = models.resnet18(weights=weights)  # Instantiate the base architecture.

    if in_channels != 3:
        # torchvision's ResNet expects 3-channel RGB by default.  If we
        # ever want to feed grayscale images or additional channels we
        # replace the stem convolution and copy or re-initialise its
        # weights accordingly.
        old_conv = model.conv1  # Original convolution layer with 3 input channels.
        bias = old_conv.bias is not None  # Remember whether the original layer had a bias term.
        model.conv1 = nn.Conv2d(
            in_channels,  # New number of input channels.
            old_conv.out_channels,  # Preserve filter count.
            kernel_size=old_conv.kernel_size,  # Keep same receptive field.
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
                    model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))  # Average RGB filters for grayscale.
                elif in_channels == 3:
                    model.conv1.weight.copy_(old_conv.weight)

    # Replace the ImageNet classification head with a CIFAR-10 head.
    in_features = model.fc.in_features  # Number of input units to the fully connected layer.
    model.fc = nn.Linear(in_features, num_classes)  # New linear layer producing 10 logits.

    return model


__all__ = ["create_resnet18"]
