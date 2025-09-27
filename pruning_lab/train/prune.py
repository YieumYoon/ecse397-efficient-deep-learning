"""Utility functions for custom unstructured and structured pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn


PruningMasks = Dict[str, torch.Tensor]


@dataclass
class PruningSummary:
    masks: PruningMasks
    global_sparsity: float
    per_parameter_sparsity: Dict[str, float]


def _iter_prunable_parameters(
    model: nn.Module,
    include_bias: bool = False,
    include_norm: bool = False,
) -> Iterable[Tuple[str, nn.Parameter]]:
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)
    for name, module in model.named_modules():
        if isinstance(module, norm_types) and not include_norm:
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if not include_bias and param_name == "bias":
                continue
            full_name = f"{name}.{param_name}" if name else param_name
            yield full_name, param


def apply_masks(model: nn.Module, masks: Optional[PruningMasks]) -> None:
    if not masks:
        return
    param_dict = dict(model.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            if name not in param_dict:
                raise KeyError(f"Mask provided for unknown parameter: {name}")
            param = param_dict[name]
            mask = mask.to(param.device, dtype=param.dtype)
            param.mul_(mask)


def _sparsity_from_masks(masks: PruningMasks) -> Tuple[int, int]:
    total = 0
    zeros = 0
    for mask in masks.values():
        total += mask.numel()
        zeros += mask.numel() - int(mask.count_nonzero().item())
    return zeros, total


def summarize_sparsity(masks: PruningMasks) -> Tuple[float, Dict[str, float]]:
    zeros, total = _sparsity_from_masks(masks)
    per_parameter = {}
    for name, mask in masks.items():
        param_total = mask.numel()
        param_zero = param_total - int(mask.count_nonzero().item())
        per_parameter[name] = param_zero / param_total if param_total else 0.0
    global_sparsity = zeros / total if total else 0.0
    return global_sparsity, per_parameter


def magnitude_unstructured_prune(
    model: nn.Module,
    amount: float,
    include_bias: bool = False,
    include_norm: bool = False,
) -> PruningSummary:
    if not 0.0 <= amount < 1.0:
        raise ValueError("Pruning amount must be in [0, 1).")

    params = list(_iter_prunable_parameters(model, include_bias, include_norm))
    if not params:
        raise ValueError("No parameters available for pruning.")

    all_scores = torch.cat([param.detach().abs().flatten() for _, param in params])
    prune_count = int(all_scores.numel() * amount)
    if prune_count == 0:
        masks = {name: torch.ones_like(param, dtype=param.dtype) for name, param in params}
        global_s, per_param = summarize_sparsity(masks)
        return PruningSummary(masks=masks, global_sparsity=global_s, per_parameter_sparsity=per_param)

    threshold = torch.topk(all_scores, prune_count, largest=False).values.max()

    masks = {}
    for name, param in params:
        mask = (param.detach().abs() > threshold).to(param.dtype)
        masks[name] = mask

    global_sparsity, per_param = summarize_sparsity(masks)
    return PruningSummary(masks=masks, global_sparsity=global_sparsity, per_parameter_sparsity=per_param)


def structured_channel_prune(
    model: nn.Module,
    amount: float,
    include_linear: bool = True,
) -> PruningSummary:
    if not 0.0 <= amount < 1.0:
        raise ValueError("Pruning amount must be in [0, 1).")

    candidate_modules = (nn.Conv2d, nn.Linear) if include_linear else (nn.Conv2d,)

    channel_scores = []
    module_meta = []
    for module_name, module in model.named_modules():
        if not isinstance(module, candidate_modules):
            continue
        weight = module.weight.detach()
        view_dims = tuple(range(1, weight.dim()))
        scores = weight.abs().mean(dim=view_dims)
        channel_scores.append(scores)
        module_meta.append((module_name, module, scores.numel()))

    if not channel_scores:
        raise ValueError("No convolutional or linear layers available for structured pruning.")

    concatenated = torch.cat(channel_scores)
    prune_count = int(concatenated.numel() * amount)
    if prune_count == 0:
        masks = {}
        for module_name, module, _ in module_meta:
            weight_mask = torch.ones_like(module.weight)
            masks[f"{module_name}.weight" if module_name else "weight"] = weight_mask
            if module.bias is not None:
                masks[f"{module_name}.bias" if module_name else "bias"] = torch.ones_like(module.bias)
        global_sparsity, per_param = summarize_sparsity(masks)
        return PruningSummary(masks=masks, global_sparsity=global_sparsity, per_parameter_sparsity=per_param)

    threshold = torch.topk(concatenated, prune_count, largest=False).values.max()

    masks: PruningMasks = {}
    for module_name, module, _ in module_meta:
        weight = module.weight.detach()
        view_dims = tuple(range(1, weight.dim()))
        scores = weight.abs().mean(dim=view_dims)
        keep_channels = scores > threshold

        if keep_channels.sum() == 0:
            max_idx = scores.argmax()
            keep_channels[max_idx] = True

        shape = [keep_channels.shape[0]] + [1] * (weight.dim() - 1)
        weight_mask = keep_channels.view(shape).to(weight.dtype)
        param_name = f"{module_name}.weight" if module_name else "weight"
        masks[param_name] = weight_mask.expand_as(weight)

        if module.bias is not None:
            bias_mask = keep_channels.to(module.bias.dtype)
            bias_name = f"{module_name}.bias" if module_name else "bias"
            masks[bias_name] = bias_mask

    global_sparsity, per_param = summarize_sparsity(masks)
    return PruningSummary(masks=masks, global_sparsity=global_sparsity, per_parameter_sparsity=per_param)


__all__ = [
    "apply_masks",
    "magnitude_unstructured_prune",
    "PruningSummary",
    "structured_channel_prune",
    "summarize_sparsity",
]
