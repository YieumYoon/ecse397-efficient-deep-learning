"""Inference helpers for evaluating saved checkpoints.

Background:
    During evaluation (also called *inference*) we are no longer
    updating model parameters.  Instead we freeze the weights, run a
    forward pass on the test set, and compute summary metrics such as
    accuracy and loss.  This separation between training and evaluation
    is important because layers like Dropout and BatchNorm behave
    differently once gradients are disabled.

Why this file matters:
    The lab rubric asks you to record pre- and post-pruning accuracies.
    Running the CLI ``test`` command under the hood calls
    ``evaluate_checkpoint`` defined here, ensuring that masks are
    re-applied before measurement so sparsity figures remain valid.
"""

# from __future__ import annotations  # Commented for Python 3.6 compatibility

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch  # Handles tensor loading and device placement.
from torch import nn
from torch.utils.data import DataLoader  # Provides batched iteration for evaluation.

from pruning_lab.train.train_loop import evaluate, EvaluationResult
from pruning_lab.train.prune import apply_masks


@dataclass
class TestResult:
    """Holds the metrics produced by ``evaluate_checkpoint``."""

    loss: float
    accuracy: float


def evaluate_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    data_loader: DataLoader,
    device: torch.device,
    strict: bool = True,
) -> TestResult:
    """Load a saved model, apply masks, and run evaluation.

    Args:
        model: Model instance created via the registry in ``main.py``.
        checkpoint_path: Location of the `.pth` file produced by the
            training or pruning commands.
        data_loader: Typically the CIFAR-10 test loader returned by
            ``get_loaders``.
        device: Torch device to run evaluation on.
        strict: Forwarded to ``load_state_dict``.  Keeping this ``True``
            ensures any key mismatches are surfaced instead of silently
            ignored.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load dictionary saved by training/pruning.
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=strict)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing keys: {missing}, unexpected keys: {unexpected}."
        )

    if checkpoint.get("masks"):
        # When evaluating pruned models we must reapply the same masks
        # that were active during fine-tuning; otherwise dormant weights
        # would spring back to life and inflate accuracy.
        apply_masks(model, checkpoint["masks"])

    model.to(device)  # Move model to CPU/GPU for inference.
    metrics: EvaluationResult = evaluate(model, data_loader, device)
    return TestResult(loss=metrics.loss, accuracy=metrics.accuracy)


__all__ = ["evaluate_checkpoint", "TestResult"]
