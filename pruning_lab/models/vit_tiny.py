"""Vision Transformer (ViT-Tiny) builders for CIFAR-10 experiments.

Background:
    Vision Transformers treat an image as a sequence of patches, similar
    to how classic Transformers process word tokens.  Each patch is
    linearly embedded, a special *class token* is prepended, and the
    sequence flows through self-attention blocks.  Because attention is
    permutation-invariant, we add positional embeddings to tell the
    model where each patch came from.

    The ViT-Tiny configuration has far fewer parameters than ViT-Base,
    making it feasible to train on CIFAR-10.  Nevertheless it still
    expects ImageNet-style preprocessors, hence the need to resize
    CIFAR-10 images.

Purpose in Lab 1:
    - Evaluate pruning on a Transformer architecture, complementing the
      CNN experiments.
    - Compare fine-tuning from large-scale pre-training (JFT/ImageNet)
      with training from scratch.

Learning tip:
    Inspect the attributes of the returned model (``model.blocks``) to
    see how attention layers are organised.  This will be useful when
    extending the structured pruning logic to operate on attention
    heads.
"""

from __future__ import annotations

from typing import Optional

import torch  # Core tensor library.
from torch import nn  # Neural network primitives.
from torch.nn import functional as F  # Additional neural network operations (interpolation, etc.).

try:
    import timm
except ImportError as exc:  # pragma: no cover - handled at runtime
    timm = None  # type: ignore


def _require_timm() -> None:
    if timm is None:
        raise ImportError(
            "timm is required for ViT models. Install via `pip install timm`."
        )


def _update_positional_embeddings(
    model: nn.Module,
    img_size: int,
    pretrained: bool,
) -> None:
    """Resize or reinitialise the positional embeddings for CIFAR-10.

    Vision Transformers segment an image into a sequence of patches and
    add positional embeddings so the attention blocks know where each
    patch came from.  When we change the input resolution (e.g. resizing
    from 224×224 to 32×32 or vice versa) the grid of patches changes
    shape.  This helper keeps the embedding tensor in sync with that
    grid by either interpolating (if we started from a pre-trained
    model) or sampling a new tensor (when training from scratch).
    """
    patch_embed = model.patch_embed  # Module that splits images into patches.
    if isinstance(patch_embed.patch_size, tuple):
        patch_h, patch_w = patch_embed.patch_size
    else:
        patch_h = patch_w = patch_embed.patch_size

    new_grid_h = img_size // patch_h  # Number of patches along the height.
    new_grid_w = img_size // patch_w  # Number of patches along the width.

    current_h, current_w = patch_embed.grid_size
    if new_grid_h == current_h and new_grid_w == current_w:
        return

    pos_embed = model.pos_embed  # Learnable positional embeddings (1 + num_patches, hidden_dim).
    cls_token = pos_embed[:, :1]  # The special classification token (kept intact).
    grid_embed = pos_embed[:, 1:]  # Positional embeddings for image patches.
    hidden_dim = grid_embed.shape[-1]  # Embedding dimension (channels).

    grid_embed = grid_embed.reshape(1, current_h, current_w, hidden_dim)  # Restore 2D grid for interpolation.
    grid_embed = grid_embed.permute(0, 3, 1, 2)  # Convert to (batch, channels, height, width) format.

    if pretrained:
        grid_embed = F.interpolate(
            grid_embed,
            size=(new_grid_h, new_grid_w),
            mode="bicubic",
            align_corners=False,
        )
    else:
        # Training from scratch means there is no useful structure in
        # the ImageNet positional embeddings, so we sample fresh values.
        grid_embed = torch.randn(1, hidden_dim, new_grid_h, new_grid_w, device=grid_embed.device)
        nn.init.trunc_normal_(grid_embed, std=0.02)  # Follow ViT initialisation scheme.

    grid_embed = grid_embed.permute(0, 2, 3, 1).reshape(1, new_grid_h * new_grid_w, hidden_dim)
    new_pos_embed = torch.cat([cls_token, grid_embed], dim=1)
    model.pos_embed = nn.Parameter(new_pos_embed)  # Replace positional embedding parameter.

    patch_embed.img_size = (img_size, img_size)  # Update metadata used inside timm.
    patch_embed.grid_size = (new_grid_h, new_grid_w)
    patch_embed.num_patches = new_grid_h * new_grid_w


def create_vit_tiny(
    num_classes: int = 10,
    pretrained: bool = True,
    img_size: int = 224,
    drop_path: float = 0.1,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """Instantiate a ViT-Tiny model using timm.

    Args:
        num_classes: Number of labels (10 for CIFAR-10).
        pretrained: Whether to start from ImageNet/JFT weights (the
            lab's "pre-trained" track) or to initialise randomly for the
            "from scratch" track.
        img_size: Target resolution to resize CIFAR-10 into before
            feeding it through the transformer.
        drop_path: Stochastic depth rate for the transformer blocks.  A
            small amount of regularisation usually helps generalisation
            when fine-tuning on CIFAR-10.
        checkpoint_path: Optional path to a custom state dict.  Useful
            when resuming experiments beyond the standard lab pipeline.

    Returns:
        A ``torch.nn.Module`` ready to be optimised by the training loop.
    """

    _require_timm()

    model = timm.create_model(
        "vit_tiny_patch16_224",  # Model identifier inside timm.
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path,  # Probability of dropping residual paths (regularisation).
    )

    if checkpoint_path:
        # Researchers may want to start from their own checkpoints.
        state = torch.load(checkpoint_path, map_location="cpu")  # Load saved state dict (weights).
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch. Missing: {missing}, unexpected: {unexpected}"
            )

    # Align the positional embeddings with the resized CIFAR grid.
    _update_positional_embeddings(model, img_size, pretrained=pretrained)

    return model


__all__ = ["create_vit_tiny"]
