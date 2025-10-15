"""ResNet-18 teacher model for knowledge distillation.

This module provides a wrapper around ResNet-18 configured for CIFAR-10.
The teacher model uses the standard ResNet-18 architecture adapted for
32×32 images with a CIFAR-10-specific stem.
"""

import torch
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock


def create_resnet18_teacher(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Create ResNet-18 teacher model for CIFAR-10.
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        pretrained: If True, use ImageNet pretrained weights (requires 224×224 input)
                   If False, train from scratch with CIFAR-10 optimized stem
    
    Returns:
        ResNet-18 model configured for CIFAR-10
    """
    if pretrained:
        # Use torchvision pretrained weights
        from torchvision import models
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        
        # Replace final FC layer for CIFAR-10
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(model.fc.bias)
        
        # Note: When using pretrained weights, input images should be 224×224
        # The dataloader will handle this resizing
    else:
        # Create ResNet-18 with CIFAR-10 optimized stem
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        
        # Replace the first conv layer for CIFAR-10 (32×32 images)
        # Use 3×3 kernel with stride=1 instead of 7×7 with stride=2
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the maxpool layer (too aggressive for 32×32 images)
        model.maxpool = nn.Identity()
        
    return model


def get_teacher_resnet_params():
    """Get recommended hyperparameters for training ResNet-18 teacher.
    
    Returns:
        dict: Training hyperparameters
    """
    return {
        'epochs': 300,
        'batch_size': 128,
        'optimizer': 'sgd',
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'scheduler': 'multistep',
        'milestones': [150, 225, 275],
        'gamma': 0.1,
    }

