"""Vision Transformer (ViT-Tiny) builders for CIFAR-10 experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

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
    patch_embed = model.patch_embed
    if isinstance(patch_embed.patch_size, tuple):
        patch_h, patch_w = patch_embed.patch_size
    else:
        patch_h = patch_w = patch_embed.patch_size

    new_grid_h = img_size // patch_h
    new_grid_w = img_size // patch_w

    current_h, current_w = patch_embed.grid_size
    if new_grid_h == current_h and new_grid_w == current_w:
        return

    pos_embed = model.pos_embed
    cls_token = pos_embed[:, :1]
    grid_embed = pos_embed[:, 1:]
    hidden_dim = grid_embed.shape[-1]

    grid_embed = grid_embed.reshape(1, current_h, current_w, hidden_dim)
    grid_embed = grid_embed.permute(0, 3, 1, 2)

    if pretrained:
        grid_embed = F.interpolate(
            grid_embed,
            size=(new_grid_h, new_grid_w),
            mode="bicubic",
            align_corners=False,
        )
    else:
        grid_embed = torch.randn(1, hidden_dim, new_grid_h, new_grid_w, device=grid_embed.device)
        nn.init.trunc_normal_(grid_embed, std=0.02)

    grid_embed = grid_embed.permute(0, 2, 3, 1).reshape(1, new_grid_h * new_grid_w, hidden_dim)
    new_pos_embed = torch.cat([cls_token, grid_embed], dim=1)
    model.pos_embed = nn.Parameter(new_pos_embed)

    patch_embed.img_size = (img_size, img_size)
    patch_embed.grid_size = (new_grid_h, new_grid_w)
    patch_embed.num_patches = new_grid_h * new_grid_w


def create_vit_tiny(
    num_classes: int = 10,
    pretrained: bool = True,
    img_size: int = 224,
    drop_path: float = 0.1,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """Instantiate a ViT-Tiny model using timm."""

    _require_timm()

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path,
    )

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch. Missing: {missing}, unexpected: {unexpected}"
            )

    _update_positional_embeddings(model, img_size, pretrained=pretrained)

    return model


__all__ = ["create_vit_tiny"]
