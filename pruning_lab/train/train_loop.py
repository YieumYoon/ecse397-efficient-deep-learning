"""Training and evaluation utilities for CIFAR-10 experiments.

Background overview:
    Training neural networks is an optimisation problem.  We take a
    model with parameters ``theta`` and minimise a loss function (for
    classification we use cross-entropy) by iteratively updating the
    weights using gradient descent.

    PyTorch breaks this process into a few standard components:

    - **Optimiser**: Implements the update rule.  Stochastic Gradient
      Descent (SGD) with momentum is a popular choice for vision tasks;
      AdamW is an alternative that adapts learning rates.
    - **Scheduler**: Adjusts the learning rate during training.  Step or
      cosine schedulers help converge faster.
    - **Mixed Precision (AMP)**: Uses 16-bit floating point math in some
      operations to speed up training on modern GPUs while keeping the
      model stable.
    - **Checkpointing**: Saves model and optimiser states so long runs
      can resume after interruption.

Connection to Lab 1:
    This file orchestrates all of the above.  Both baseline training and
    fine-tuning after pruning call into ``train_model``.  Understanding
    each step here means you can reproduce the experiments manually on a
    different dataset or architecture later on.
"""

from __future__ import annotations

import json  # For writing metrics/history to disk.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch  # Tensor library with autograd.
from torch import nn  # Neural network modules (loss functions, etc.).
from torch.cuda import amp  # Automatic Mixed Precision utilities.
from torch.utils.data import DataLoader  # Provides batched data iteration.

from .prune import apply_masks, summarize_sparsity


@dataclass
class OptimizerConfig:
    """Configuration bundle for creating an optimiser."""

    name: str = "sgd"  # Choice of optimiser algorithm.
    lr: float = 0.1  # Learning rate (step size).
    momentum: float = 0.9  # Momentum factor for SGD.
    weight_decay: float = 5e-4  # L2 regularisation strength.
    betas: Tuple[float, float] = (0.9, 0.999)  # AdamW momentum coefficients.


@dataclass
class SchedulerConfig:
    """Parameters for optional learning rate schedulers."""

    name: Optional[str] = None  # Scheduler identifier.
    milestones: Tuple[int, ...] = (150, 225)  # Epochs to decay LR for MultiStep scheduler.
    gamma: float = 0.1  # Multiplicative factor applied at milestones.
    t_max: int = 200  # Period for cosine scheduler.


@dataclass
class TrainConfig:
    """Settings that control a training run in the lab."""

    epochs: int = 200  # Number of passes over training data.
    grad_clip: Optional[float] = None  # Max gradient norm for clipping (None disables).
    amp: bool = False  # Enable automatic mixed precision if True.
    log_interval: int = 50  # (Reserved for future logging; not used explicitly here.)
    eval_interval: int = 1  # Evaluate on validation set every N epochs.
    output_dir: Path = Path("models_saved")  # Directory to store checkpoints.
    checkpoint_name: str = "model.pth"  # Filename for latest checkpoint.
    resume_path: Optional[Path] = None  # Path to checkpoint for resuming training.
    metrics_path: Optional[Path] = None  # Optional file to save JSON history.


@dataclass
class EpochMetrics:
    """Simple container for loss and accuracy numbers."""

    loss: float  # Average loss over the epoch.
    accuracy: float  # Classification accuracy for the epoch.


@dataclass
class TrainingHistoryEntry:
    """Stores metrics for a single epoch."""

    epoch: int  # Epoch index (0-based).
    train: EpochMetrics  # Training metrics.
    val: Optional[EpochMetrics]  # Validation metrics (None if not evaluated).


@dataclass
class TrainingSummary:
    """Final report returned by ``train_model``.

    When you run the CLI, the summary is converted to JSON so you can
    track progress and retrieve best checkpoints after the job
    completes.
    """

    history: List[TrainingHistoryEntry] = field(default_factory=list)  # Per-epoch metrics ledger.
    best_accuracy: float = 0.0  # Highest validation accuracy observed.
    best_epoch: int = -1  # Epoch index when best accuracy occurred.
    best_checkpoint: Optional[Path] = None  # Path to checkpoint capturing best accuracy.
    final_checkpoint: Optional[Path] = None  # Path to latest checkpoint at loop end.
    sparsity: Optional[float] = None  # Global sparsity when masks are applied.


@dataclass
class EvaluationResult:
    """Return type for evaluation loops."""

    loss: float
    accuracy: float


def set_seed(seed: int) -> None:
    """Set deterministic seeds so experiments are reproducible."""
    torch.manual_seed(seed)  # Deterministic CPU computations.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Deterministic GPU computations across devices.


def _create_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Factory for standard optimisers used in the lab."""
    if config.name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    if config.name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.name}")


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Optional[SchedulerConfig],
    total_epochs: int,
):
    """Instantiate an LR scheduler if requested."""
    if scheduler_config is None or scheduler_config.name is None:
        return None
    name = scheduler_config.name.lower()
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(scheduler_config.milestones),
            gamma=scheduler_config.gamma,
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.t_max or total_epochs,
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> EvaluationResult:
    """Run a forward pass over a dataset without updating weights."""
    model.eval()  # Switch layers like BatchNorm/Dropout to evaluation mode.
    criterion = criterion or nn.CrossEntropyLoss()
    running_loss = 0.0  # Accumulates loss over all samples.
    correct = 0  # Number of correctly classified samples.
    total = 0  # Total number of evaluated samples.

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)  # Move batch to GPU/CPU.
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)  # Scale by batch size for averaging later.
            _, predicted = outputs.max(1)  # Predicted class index per sample.
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  # Count matches with ground truth.

    avg_loss = running_loss / total
    accuracy = correct / total
    return EvaluationResult(loss=avg_loss, accuracy=accuracy)


def _step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: amp.GradScaler,
    use_amp: bool,
    grad_clip: Optional[float],
    masks: Optional[Dict[str, torch.Tensor]],
) -> Tuple[float, torch.Tensor]:
    """Perform one optimisation step and reapply pruning masks."""
    optimizer.zero_grad(set_to_none=True)  # Clear gradients from previous step.
    with amp.autocast(enabled=use_amp):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()  # Backpropagate with optional mixed precision scaling.
    if grad_clip is not None:
        scaler.unscale_(optimizer)  # Bring gradients back to FP32 before clipping.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)  # Apply optimiser update.
    scaler.update()  # Adjust scaling factor for next iteration.
    apply_masks(model, masks)  # Enforce sparsity constraints.
    return loss.item(), outputs.detach()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer_config: OptimizerConfig,
    train_config: TrainConfig,
    device: torch.device,
    scheduler_config: Optional[SchedulerConfig] = None,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> TrainingSummary:
    """Full training loop that the CLI and notebooks call into.

    The loop is intentionally explicit: gather metrics, checkpoint at
    the end of every epoch, optionally resume, and reapply pruning masks
    after each optimisation step.  This scaffolding mirrors what you
    would build by hand when running experiments on a remote server.
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = _create_optimizer(model, optimizer_config)
    scheduler = _create_scheduler(optimizer, scheduler_config, train_config.epochs)
    scaler = amp.GradScaler(enabled=train_config.amp)  # Handles scaling for mixed precision.

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / train_config.checkpoint_name  # Path to latest checkpoint.
    history: List[TrainingHistoryEntry] = []
    best_acc = 0.0
    best_epoch = -1
    best_checkpoint = None

    start_epoch = 0
    if train_config.resume_path:
        start_epoch, checkpoint_best, checkpoint_masks = _load_checkpoint(
            path=train_config.resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
        )
        if masks is None and checkpoint_masks:
            masks = checkpoint_masks  # Reuse masks stored in checkpoint if caller did not provide any.
        best_acc = checkpoint_best

    for epoch in range(start_epoch, train_config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss, outputs = _step(
                model,
                inputs,
                targets,
                optimizer,
                criterion,
                scaler,
                train_config.amp,
                train_config.grad_clip,
                masks,
            )
            running_loss += loss * inputs.size(0)

            # Reuse the outputs computed during the optimisation step to
            # avoid another forward pass.
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        val_metrics = None
        if val_loader is not None and ((epoch + 1) % train_config.eval_interval == 0):
            val_metrics = evaluate(model, val_loader, device, criterion)
            if val_metrics.accuracy > best_acc:
                best_acc = val_metrics.accuracy
                best_epoch = epoch
                best_checkpoint = output_dir / f"best_{train_config.checkpoint_name}"
                _save_checkpoint(best_checkpoint, model, optimizer, scaler, epoch, best_acc, masks)

        history.append(
            TrainingHistoryEntry(
                epoch=epoch,
                train=EpochMetrics(loss=train_loss, accuracy=train_acc),
                val=EpochMetrics(loss=val_metrics.loss, accuracy=val_metrics.accuracy)
                if val_metrics
                else None,
            )
        )

        # Always save the latest checkpoint so jobs interrupted on a
        # remote server can resume with ``--resume`` next time.
        _save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, best_acc, masks)
        if scheduler:
            scheduler.step()

    apply_masks(model, masks)
    final_sparsity = None
    if masks:
        sparsity, _ = summarize_sparsity(masks)
        final_sparsity = sparsity

    summary = TrainingSummary(
        history=history,
        best_accuracy=best_acc,
        best_epoch=best_epoch,
        best_checkpoint=best_checkpoint,
        final_checkpoint=ckpt_path,
        sparsity=final_sparsity,
    )

    if train_config.metrics_path:
        _export_history(train_config.metrics_path, summary)

    return summary


def _export_history(path: Path, summary: TrainingSummary) -> None:
    """Serialize the training history to JSON for later analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "history": [
            {
                "epoch": entry.epoch,
                "train_loss": entry.train.loss,
                "train_accuracy": entry.train.accuracy,
                "val_loss": entry.val.loss if entry.val else None,
                "val_accuracy": entry.val.accuracy if entry.val else None,
            }
            for entry in summary.history
        ],
        "best_accuracy": summary.best_accuracy,
        "best_epoch": summary.best_epoch,
        "best_checkpoint": str(summary.best_checkpoint) if summary.best_checkpoint else None,
        "final_checkpoint": str(summary.final_checkpoint) if summary.final_checkpoint else None,
        "sparsity": summary.sparsity,
    }
    path.write_text(json.dumps(data, indent=2))


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    epoch: int,
    best_acc: float,
    masks: Optional[Dict[str, torch.Tensor]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Persist everything required to resume training: model weights,
    # optimiser/scheduler state, AMP scaler, and pruning masks.
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_accuracy": best_acc,
            "masks": masks,
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
) -> Tuple[int, float, Optional[Dict[str, torch.Tensor]]]:
    """Restore a checkpoint created by ``_save_checkpoint``."""
    checkpoint = torch.load(path, map_location="cpu")  # Load dictionary containing training state.
    model.load_state_dict(checkpoint["model_state"])  # Restore model weights.
    optimizer.load_state_dict(checkpoint["optimizer_state"])  # Restore optimiser state (momenta, etc.).
    scaler.load_state_dict(checkpoint.get("scaler_state", {}))  # Mixed precision metadata (safe to be empty).
    masks = checkpoint.get("masks")
    if masks:
        apply_masks(model, masks)
    next_epoch = checkpoint.get("epoch", 0) + 1  # Resume from the epoch after the saved one.
    best_acc = checkpoint.get("best_accuracy", 0.0)
    return next_epoch, best_acc, masks


__all__ = [
    "EvaluationResult",
    "EpochMetrics",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainConfig",
    "TrainingHistoryEntry",
    "TrainingSummary",
    "evaluate",
    "set_seed",
    "train_model",
]
