"""ResNet-8 student model for knowledge distillation.

This module implements a smaller ResNet variant with [1, 1, 1, 1] basic blocks
(1 block per stage instead of 2). This creates a compact student model that
can learn from the ResNet-18 teacher via knowledge distillation.
"""

import torch
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock


def create_resnet8_student(num_classes: int = 10) -> nn.Module:
    """Create ResNet-8 student model for CIFAR-10.
    
    ResNet-8 has significantly fewer layers than ResNet-18:
    - Stage 1: 1 BasicBlock (2 conv layers)
    - Stage 2: 1 BasicBlock (2 conv layers)
    - Stage 3: 1 BasicBlock (2 conv layers)
    - Stage 4: 1 BasicBlock (2 conv layers)
    Total: ~8 conv layers vs ResNet-18's 16 conv layers
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
    
    Returns:
        ResNet-8 student model optimized for CIFAR-10
    """
    # Create ResNet with [1, 1, 1, 1] configuration
    # Each BasicBlock has 2 conv layers, so [1, 1, 1, 1] = 8 conv layers
    model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
    
    # Adapt stem for CIFAR-10 (32×32 images)
    # Use 3×3 kernel with stride=1 instead of 7×7 with stride=2
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the maxpool layer (too aggressive for small 32×32 images)
    model.maxpool = nn.Identity()
    
    return model


def get_student_resnet_params():
    """Get recommended hyperparameters for training ResNet-8 student.
    
    Returns:
        dict: Training hyperparameters
    """
    return {
        'epochs': 200,
        'batch_size': 128,
        'optimizer': 'sgd',
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'scheduler': 'multistep',
        'milestones': [100, 150, 180],
        'gamma': 0.1,
    }


def get_distillation_params():
    """Get recommended hyperparameters for distilling ResNet-8.
    
    Returns:
        dict: Distillation hyperparameters
    """
    return {
        'epochs': 200,
        'batch_size': 128,
        'optimizer': 'sgd',
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'scheduler': 'multistep',
        'milestones': [100, 150, 180],
        'gamma': 0.1,
        'alpha': 0.5,        # Weight for hard targets
        'temperature': 4.0,  # Temperature for soft targets
    }

