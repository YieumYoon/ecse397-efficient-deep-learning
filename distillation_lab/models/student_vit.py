"""ViT-Student model for knowledge distillation.

This module implements a smaller Vision Transformer with 6 layers and
192 embedding dimensions (vs ViT-Tiny's 12 layers and 192/384 dims).
"""

import torch
from torch import nn


def create_vit_student(
    num_classes: int = 10,
    img_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 192,
    depth: int = 6,
    num_heads: int = 3,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Create ViT-Student model for CIFAR-10.
    
    A smaller Vision Transformer designed to learn from ViT-Tiny teacher:
    - 6 transformer layers (vs ViT-Tiny's 12)
    - 192 embedding dimensions
    - 3 attention heads (vs ViT-Tiny's 3)
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        img_size: Input image size (default: 224)
        patch_size: Size of image patches (default: 16)
        embed_dim: Embedding dimension (default: 192)
        depth: Number of transformer layers (default: 6)
        num_heads: Number of attention heads (default: 3)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        drop_rate: Dropout rate (default: 0.0)
        attn_drop_rate: Attention dropout rate (default: 0.0)
        drop_path_rate: Stochastic depth rate (default: 0.0)
    
    Returns:
        ViT-Student model configured for CIFAR-10
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm library required for ViT models. Install with: pip install timm")
    
    # Create a custom ViT with reduced depth and dimensions
    # We'll use timm's VisionTransformer class directly
    from timm.models.vision_transformer import VisionTransformer
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=nn.LayerNorm,
    )
    
    return model


def get_student_vit_params():
    """Get recommended hyperparameters for training ViT-Student.
    
    Returns:
        dict: Training hyperparameters
    """
    return {
        'epochs': 150,
        'batch_size': 128,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.05,
        'scheduler': 'cosine',
        't_max': 150,
        'img_size': 224,
    }


def get_distillation_params():
    """Get recommended hyperparameters for distilling ViT-Student.
    
    Returns:
        dict: Distillation hyperparameters
    """
    return {
        'epochs': 150,
        'batch_size': 128,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.05,
        'scheduler': 'cosine',
        't_max': 150,
        'alpha': 0.5,        # Weight for hard targets
        'temperature': 4.0,  # Temperature for soft targets
        'img_size': 224,
    }

