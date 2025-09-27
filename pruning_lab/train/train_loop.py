"""Training and evaluation utilities for CIFAR-10 experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader

from .prune import apply_masks, summarize_sparsity


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    name: Optional[str] = None
    milestones: Tuple[int, ...] = (150, 225)
    gamma: float = 0.1
    t_max: int = 200


@dataclass
class TrainConfig:
    epochs: int = 200
    grad_clip: Optional[float] = None
    amp: bool = False
    log_interval: int = 50
    eval_interval: int = 1
    output_dir: Path = Path("models_saved")
    checkpoint_name: str = "model.pth"
    resume_path: Optional[Path] = None
    metrics_path: Optional[Path] = None


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass
class TrainingHistoryEntry:
    epoch: int
    train: EpochMetrics
    val: Optional[EpochMetrics]


@dataclass
class TrainingSummary:
    history: List[TrainingHistoryEntry] = field(default_factory=list)
    best_accuracy: float = 0.0
    best_epoch: int = -1
    best_checkpoint: Optional[Path] = None
    final_checkpoint: Optional[Path] = None
    sparsity: Optional[float] = None


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
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
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

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
    optimizer.zero_grad(set_to_none=True)
    with amp.autocast(enabled=use_amp):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    if grad_clip is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    apply_masks(model, masks)
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
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = _create_optimizer(model, optimizer_config)
    scheduler = _create_scheduler(optimizer, scheduler_config, train_config.epochs)
    scaler = amp.GradScaler(enabled=train_config.amp)

    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / train_config.checkpoint_name
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
            masks = checkpoint_masks
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
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint.get("scaler_state", {}))
    masks = checkpoint.get("masks")
    if masks:
        apply_masks(model, masks)
    next_epoch = checkpoint.get("epoch", 0) + 1
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
