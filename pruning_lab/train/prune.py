"""Utility functions for custom unstructured and structured pruning.

Background concepts:
    - **Pruning** removes weights or entire channels from a neural
      network to make it sparser (i.e., contains more zeros).  Sparse
      networks can store fewer parameters and sometimes execute faster.
    - **Unstructured pruning** removes individual scalar weights.  This
      often yields extremely high sparsity but standard hardware does
      not always exploit it efficiently without specialised libraries.
    - **Structured pruning** removes larger units (convolutional
      filters, attention heads, etc.).  The resulting model has smaller
      dense layers that translate to real speedups after retraining.

Why the lab disallows ``torch.nn.utils.prune``:
    Implementing masks manually deepens your understanding of how model
    parameters are stored, how to zero them out safely, and how sparsity
    propagates through training.  The same knowledge transfers to
    advanced research topics like lottery-ticket hypotheses and model
    compression.

Reading guide:
    - Start with ``magnitude_unstructured_prune`` to see how global
      thresholds are derived from absolute weight values.
    - Move to ``structured_channel_prune`` to understand how per-channel
      averages determine which filters survive.
    - Observe how ``apply_masks`` is invoked after every optimiser step
      in ``train_loop.py``—this is crucial to keep pruned weights at zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch  # Tensor library and numerics.
from torch import nn  # Neural network modules (Conv2d, Linear, etc.).


PruningMasks = Dict[str, torch.Tensor]  # Maps parameter names to binary (0/1) masks.


@dataclass
class PruningSummary:
    """Container for the results of a pruning pass.

    Attributes:
        masks: Dictionary mapping parameter names to binary masks.  Each
            mask has the same shape as the parameter it should zero out.
        global_sparsity: Fraction of weights set to zero across the
            entire model.  This feeds directly into the lab's reporting
            requirements (e.g. "achieve ≥70% sparsity").
        per_parameter_sparsity: Layer-wise sparsity breakdown so you can
            diagnose which layers pruned more aggressively.
    """

    masks: PruningMasks
    global_sparsity: float
    per_parameter_sparsity: Dict[str, float]


def _iter_prunable_parameters(
    model: nn.Module,
    include_bias: bool = False,
    include_norm: bool = False,
) -> Iterable[Tuple[str, nn.Parameter]]:
    """Yield parameters that we're allowed to prune.

    Args:
        model: Neural network under consideration.
        include_bias: Bias vectors are small but you may choose to prune
            them during unstructured pruning for higher sparsity.
        include_norm: BatchNorm and LayerNorm parameters are often kept
            untouched because they stabilise training.  When the lab
            asks you to experiment, toggling this flag lets you explore
            both avenues.
    """
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)
    for name, module in model.named_modules():
        if isinstance(module, norm_types) and not include_norm:
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if not include_bias and param_name == "bias":
                continue
            full_name = f"{name}.{param_name}" if name else param_name  # Construct dotted parameter name.
            yield full_name, param


def apply_masks(model: nn.Module, masks: Optional[PruningMasks]) -> None:
    """Zero out weights in-place according to the provided masks.

    This mimics how weight sparsity is enforced during pruning.  After
    each optimiser step we immediately re-apply the mask so gradients do
    not regrow the pruned weights.
    """
    if not masks:
        return
    param_dict = dict(model.named_parameters())  # Look-up table from parameter names to tensors.
    with torch.no_grad():
        for name, mask in masks.items():
            if name not in param_dict:
                raise KeyError(f"Mask provided for unknown parameter: {name}")
            param = param_dict[name]
            mask = mask.to(param.device, dtype=param.dtype)  # Ensure mask lives on same device + dtype.
            param.mul_(mask)  # In-place multiply-by-zero to prune weights.


def _sparsity_from_masks(masks: PruningMasks) -> Tuple[int, int]:
    """Count zeroed weights (numerator) and total weights (denominator)."""
    total = 0  # Total number of weights covered by masks.
    zeros = 0  # Count of weights that are pruned (mask value 0).
    for mask in masks.values():
        total += mask.numel()
        zeros += mask.numel() - int(mask.count_nonzero().item())
    return zeros, total


def summarize_sparsity(masks: PruningMasks) -> Tuple[float, Dict[str, float]]:
    """Compute global and per-parameter sparsity ratios.

    Returns:
        Tuple consisting of the overall sparsity and a dictionary of
        individual parameter sparsities.  These values are required when
        filling out ``report.json`` at the end of the lab.
    """
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
    """Perform classic magnitude-based unstructured pruning.

    Strategy:
        1. Collect every prunable weight in the network.
        2. Compute their absolute values (importance scores).
        3. Determine a global threshold so the desired ``amount`` of
           weights become zero.
        4. Build binary masks and return them together with sparsity
           metrics.

    Unstructured pruning is effective for maximising sparsity but does
    not immediately translate to hardware speedups.  Nevertheless, it
    is required by the lab and provides insight into which weights are
    less important.
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError("Pruning amount must be in [0, 1).")

    params = list(_iter_prunable_parameters(model, include_bias, include_norm))
    if not params:
        raise ValueError("No parameters available for pruning.")

    all_scores = torch.cat([param.detach().abs().flatten() for _, param in params])  # Collect importance scores.
    prune_count = int(all_scores.numel() * amount)  # Number of weights to remove.
    if prune_count == 0:
        masks = {name: torch.ones_like(param, dtype=param.dtype) for name, param in params}
        global_s, per_param = summarize_sparsity(masks)
        return PruningSummary(masks=masks, global_sparsity=global_s, per_parameter_sparsity=per_param)

    # The kth smallest magnitude becomes our cut-off.  Any weight with
    # absolute value below or equal to this threshold will be pruned.
    threshold = torch.topk(all_scores, prune_count, largest=False).values.max()

    masks = {}
    for name, param in params:
        mask = (param.detach().abs() > threshold).to(param.dtype)  # 1 keeps weight, 0 prunes it.
        masks[name] = mask

    global_sparsity, per_param = summarize_sparsity(masks)
    return PruningSummary(masks=masks, global_sparsity=global_sparsity, per_parameter_sparsity=per_param)


def structured_channel_prune(
    model: nn.Module,
    amount: float,
    include_linear: bool = True,
) -> PruningSummary:
    """Channel-wise structured pruning used for hardware-friendly sparsity.

    Instead of pruning individual scalar weights, structured pruning
    removes entire convolutional filters or linear neurons.  This
    reduces both parameter count and, after re-exporting/compiling the
    network, can lead to real inference-time acceleration.
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError("Pruning amount must be in [0, 1).")

    candidate_modules = (nn.Conv2d, nn.Linear) if include_linear else (nn.Conv2d,)  # Layers eligible for channel pruning.

    channel_scores = []  # Stores per-channel importance values for all modules.
    module_meta = []  # Keeps track of module objects and names.
    for module_name, module in model.named_modules():
        if not isinstance(module, candidate_modules):
            continue
        weight = module.weight.detach()  # Weight tensor (out_channels, ...).
        view_dims = tuple(range(1, weight.dim()))  # Dimensions to average over (everything except output channel).
        scores = weight.abs().mean(dim=view_dims)  # Average magnitude per output channel.
        channel_scores.append(scores)
        module_meta.append((module_name, module, scores.numel()))  # Track metadata for reconstruction later.

    if not channel_scores:
        raise ValueError("No convolutional or linear layers available for structured pruning.")

    concatenated = torch.cat(channel_scores)  # Global list of channel scores.
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

    # Same approach as in unstructured pruning: determine a magnitude
    # threshold that drops the desired fraction of channels globally.
    threshold = torch.topk(concatenated, prune_count, largest=False).values.max()

    masks: PruningMasks = {}
    for module_name, module, _ in module_meta:
        weight = module.weight.detach()
        view_dims = tuple(range(1, weight.dim()))
        scores = weight.abs().mean(dim=view_dims)
        keep_channels = scores > threshold  # Boolean mask over output channels.

        if keep_channels.sum() == 0:
            max_idx = scores.argmax()
            keep_channels[max_idx] = True

        shape = [keep_channels.shape[0]] + [1] * (weight.dim() - 1)  # Reshape for broadcasting.
        weight_mask = keep_channels.view(shape).to(weight.dtype)
        param_name = f"{module_name}.weight" if module_name else "weight"
        masks[param_name] = weight_mask.expand_as(weight)  # Broadcast mask to match weight tensor.

        if module.bias is not None:
            bias_mask = keep_channels.to(module.bias.dtype)  # Apply same keep/drop decision to bias vector.
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
