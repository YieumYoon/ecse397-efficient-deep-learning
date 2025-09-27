"""Inference helpers for evaluating saved checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..train.train_loop import evaluate, EvaluationResult
from ..train.prune import apply_masks


@dataclass
class TestResult:
    loss: float
    accuracy: float


def evaluate_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    data_loader: DataLoader,
    device: torch.device,
    strict: bool = True,
) -> TestResult:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=strict)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing keys: {missing}, unexpected keys: {unexpected}."
        )

    if checkpoint.get("masks"):
        apply_masks(model, checkpoint["masks"])

    model.to(device)
    metrics: EvaluationResult = evaluate(model, data_loader, device)
    return TestResult(loss=metrics.loss, accuracy=metrics.accuracy)


__all__ = ["evaluate_checkpoint", "TestResult"]
