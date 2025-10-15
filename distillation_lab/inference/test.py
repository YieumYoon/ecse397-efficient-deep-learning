"""Inference helpers for evaluating saved checkpoints.

Background:
    During evaluation (also called *inference*) we are no longer
    updating model parameters.  Instead we freeze the weights, run a
    forward pass on the test set, and compute summary metrics such as
    accuracy and loss.  This separation between training and evaluation
    is important because layers like Dropout and BatchNorm behave
    differently once gradients are disabled.

Why this file matters:
    The lab rubric asks you to report teacher and student accuracies.
    ``evaluate_checkpoint`` defined here loads a saved checkpoint and
    evaluates it on a provided ``DataLoader``. If the checkpoint
    contains optional pruning masks under the ``"masks"`` key, those
    masks are applied before evaluation to ensure consistency.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TestResult:
    """Holds the metrics produced by ``evaluate_checkpoint``."""

    loss: float
    accuracy: float


def _evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Minimal evaluation loop with CrossEntropy loss.

    Returns average loss and accuracy in percent.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def _apply_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Apply binary masks to matching model parameters in-place.

    If a mask name or shape does not match a parameter, the mask is skipped.
    """
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mask = masks.get(name)
        if mask is None:
            continue
        if mask.shape != param.data.shape:
            # Shape mismatch: skip
            continue
        param.data.mul_(mask.to(device))


def evaluate_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    data_loader: DataLoader,
    device: torch.device,
    strict: bool = True,
) -> TestResult:
    """Load a saved model, optionally apply masks, and run evaluation.

    Args:
        model: Model instance created via the registry in ``main.py``.
        checkpoint_path: Path to the `.pth` checkpoint produced by training.
        data_loader: Typically the CIFAR-10 test loader returned by ``get_loaders``.
        device: Torch device to run evaluation on.
        strict: Forwarded to ``load_state_dict``.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Resolve state_dict location
    state_dict = None
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif all(isinstance(k, str) for k in checkpoint.keys()):
            # Might already be a raw state_dict
            state_dict = checkpoint
    else:
        # Unexpected format
        raise RuntimeError("Unsupported checkpoint format: expected dict")

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing keys: {missing}, unexpected keys: {unexpected}."
        )

    # Apply masks if present (optional)
    masks = checkpoint.get("masks") if isinstance(checkpoint, dict) else None
    if masks:
        _apply_masks(model, masks)

    model.to(device)
    avg_loss, accuracy = _evaluate(model, data_loader, device)
    return TestResult(loss=avg_loss, accuracy=accuracy)


__all__ = ["evaluate_checkpoint", "TestResult"]
