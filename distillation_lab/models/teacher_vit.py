"""ViT-Tiny teacher model for knowledge distillation.

This module provides a wrapper around ViT-Tiny configured for CIFAR-10.
The teacher uses a pretrained Vision Transformer adapted for image classification.
"""

import torch
from torch import nn


def create_vit_tiny_teacher(num_classes: int = 10, img_size: int = 224, pretrained: bool = True) -> nn.Module:
    """Create ViT-Tiny teacher model for CIFAR-10.
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        img_size: Input image size (default: 224 for pretrained ViT)
        pretrained: If True, use pretrained weights and fine-tune
    
    Returns:
        ViT-Tiny model configured for CIFAR-10
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm library required for ViT models. Install with: pip install timm")
    
    if pretrained:
        # Load pretrained ViT-Tiny from timm
        # vit_tiny_patch16_224 has 12 layers, 192 embedding dim, 3 heads
        model = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=True,
            num_classes=num_classes,
            img_size=img_size,
        )
    else:
        # Create ViT-Tiny from scratch
        model = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            num_classes=num_classes,
            img_size=img_size,
        )
    
    return model


def get_teacher_vit_params():
    """Get recommended hyperparameters for training ViT-Tiny teacher.
    
    Returns:
        dict: Training hyperparameters
    """
    return {
        'epochs': 100,
        'batch_size': 128,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.05,
        'scheduler': 'cosine',
        't_max': 100,
        'img_size': 224,
    }

