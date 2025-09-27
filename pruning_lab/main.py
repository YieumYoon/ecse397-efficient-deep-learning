"""Command-line entry point for training, pruning, and evaluating models.

Background:
    Command-line interfaces (CLIs) are common in machine learning
    research because they make experiments reproducible.  Instead of
    editing Python files each time you change a hyperparameter, you pass
    arguments such as ``--lr`` or ``--epochs`` directly to a script.  The
    standard workflow looks like this:

    .. code-block:: bash

        python -m pruning_lab.main train --model resnet18 --epochs 200

    The ``-m`` flag tells Python to execute a module.  Inside this file
    we parse the arguments, construct the appropriate model and
    DataLoader, and call the training/pruning/evaluation helpers defined
    elsewhere in the repository.

Lab connection:
    The assignment requires multiple steps (train, prune, evaluate).  By
    understanding the CLI you can run these steps on any machine—from
    your laptop to a school server—while keeping a record of exactly
    which flags were used.
"""

from __future__ import annotations

import argparse  # Builds the command-line interface.
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from pruning_lab.data.dataloader import get_loaders
from pruning_lab.inference.test import evaluate_checkpoint
from pruning_lab.models.resnet18 import create_resnet18
from pruning_lab.models.vit_tiny import create_vit_tiny
from pruning_lab.train.prune import (
    apply_masks,
    magnitude_unstructured_prune,
    structured_channel_prune,
    summarize_sparsity,
)
from pruning_lab.train.train_loop import (
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    train_model,
)


@dataclass
class ModelSpec:
    name: str  # Identifier exposed to the CLI.
    builder: Callable[..., torch.nn.Module]  # Function that returns an instantiated model.
    default_img_size: int  # Image size expected by the model (used for dataloaders).
    default_pretrained: bool  # Whether pretrained weights are loaded by default.
    force_pretrained: bool = False  # If True, CLI cannot disable pretraining.


MODEL_SPECS: Dict[str, ModelSpec] = {
    "resnet18": ModelSpec(
        name="resnet18",
        builder=lambda pretrained, num_classes, img_size, **kwargs: create_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
        ),
        default_img_size=32,
        default_pretrained=False,
    ),
    "vit_tiny_pretrained": ModelSpec(
        name="vit_tiny_pretrained",
        builder=lambda pretrained, num_classes, img_size, **kwargs: create_vit_tiny(
            num_classes=num_classes,
            pretrained=True,
            img_size=img_size,
            drop_path=kwargs.get("drop_path", 0.1),
        ),
        default_img_size=224,
        default_pretrained=True,
        force_pretrained=True,
    ),
    "vit_tiny_scratch": ModelSpec(
        name="vit_tiny_scratch",
        builder=lambda pretrained, num_classes, img_size, **kwargs: create_vit_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
            img_size=img_size,
            drop_path=kwargs.get("drop_path", 0.1),
        ),
        default_img_size=224,
        default_pretrained=False,
    ),
}


def _parse_args() -> argparse.Namespace:
    """Define the CLI surface and parse user-provided arguments."""
    parser = argparse.ArgumentParser(description="Efficient pruning lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)  # Parent parser for subcommands.

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model from scratch or fine-tune")
    _add_model_arguments(train_parser, include_pretrained_flag=True)
    train_parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    train_parser.add_argument("--lr", type=float, default=0.1, help="Base learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    train_parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    train_parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adamw"],
        help="Optimizer to use",
    )
    train_parser.add_argument(
        "--scheduler",
        type=str,
        default="multistep",
        choices=["none", "multistep", "cosine"],
        help="Learning rate scheduler",
    )
    train_parser.add_argument(
        "--milestones",
        type=int,
        nargs="*",
        default=[150, 225],
        help="Milestones for MultiStepLR scheduler",
    )
    train_parser.add_argument("--gamma", type=float, default=0.1, help="LR decay factor")
    train_parser.add_argument("--t-max", type=int, default=200, help="T_max for cosine annealing")
    train_parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping value")
    train_parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    train_parser.add_argument("--data-dir", type=str, default=None, help="Custom data directory")
    train_parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    train_parser.add_argument("--no-pin", action="store_true", help="Disable pinned memory")
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="pruning_lab/models_saved",
        help="Directory to store checkpoints",
    )
    train_parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Custom checkpoint filename (defaults to <model>.pth)",
    )
    train_parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    train_parser.add_argument("--metrics-path", type=str, default=None, help="Optional JSON metrics output")
    train_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    train_parser.add_argument("--drop-path", type=float, default=0.1, help="Drop path rate (ViT only)")
    train_parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Load initial weights before training",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Evaluate a trained model checkpoint")
    _add_model_arguments(test_parser, include_pretrained_flag=False)
    test_parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path to evaluate")
    test_parser.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size")
    test_parser.add_argument("--data-dir", type=str, default=None, help="Custom data directory")
    test_parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    test_parser.add_argument("--no-pin", action="store_true", help="Disable pinned memory")
    test_parser.add_argument("--strict", action="store_true", help="Enforce strict state dict loading")

    # Prune command
    prune_parser = subparsers.add_parser("prune", help="Apply pruning and optionally fine-tune")
    _add_model_arguments(prune_parser, include_pretrained_flag=False)
    prune_parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to prune")
    prune_parser.add_argument(
        "--prune-type",
        type=str,
        choices=["unstructured", "structured"],
        required=True,
        help="Pruning strategy",
    )
    prune_parser.add_argument("--amount", type=float, required=True, help="Fraction of weights/channels to prune")
    prune_parser.add_argument("--include-bias", action="store_true", help="Include bias parameters during pruning")
    prune_parser.add_argument(
        "--include-norm",
        action="store_true",
        help="Include normalization layers in unstructured pruning",
    )
    prune_parser.add_argument("--finetune-epochs", type=int, default=0, help="Fine-tuning epochs after pruning")
    prune_parser.add_argument("--batch-size", type=int, default=128, help="Fine-tuning batch size")
    prune_parser.add_argument("--lr", type=float, default=0.01, help="Fine-tuning learning rate")
    prune_parser.add_argument("--weight-decay", type=float, default=5e-4, help="Fine-tuning weight decay")
    prune_parser.add_argument("--momentum", type=float, default=0.9, help="Fine-tuning momentum")
    prune_parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"], help="Fine-tuning optimizer")
    prune_parser.add_argument("--scheduler", type=str, default="none", choices=["none", "multistep", "cosine"], help="Fine-tuning scheduler")
    prune_parser.add_argument("--milestones", type=int, nargs="*", default=[10, 20], help="Scheduler milestones")
    prune_parser.add_argument("--gamma", type=float, default=0.2, help="Scheduler gamma")
    prune_parser.add_argument("--t-max", type=int, default=50, help="Cosine scheduler T_max")
    prune_parser.add_argument("--eval-interval", type=int, default=1, help="Fine-tuning evaluation interval")
    prune_parser.add_argument("--amp", action="store_true", help="Use AMP during fine-tuning")
    prune_parser.add_argument("--data-dir", type=str, default=None, help="Custom data directory")
    prune_parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    prune_parser.add_argument("--no-pin", action="store_true", help="Disable pinned memory")
    prune_parser.add_argument(
        "--output-checkpoint",
        type=str,
        default=None,
        help="Destination for pruned checkpoint (defaults to models_saved/<model>_<type>.pth)",
    )
    prune_parser.add_argument("--drop-path", type=float, default=0.1, help="Drop path for ViT")

    return parser.parse_args()


def _add_model_arguments(parser: argparse.ArgumentParser, include_pretrained_flag: bool) -> None:
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_SPECS.keys()),
        required=True,
        help="Model architecture",
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of target classes")
    parser.add_argument("--img-size", type=int, default=None, help="Override default image size")
    parser.add_argument("--device", type=str, default=None, help="Torch device (defaults to cuda if available)")
    if include_pretrained_flag:
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            action="store_true",
            help="Use pretrained weights when available",
        )
        parser.add_argument(
            "--no-pretrained",
            dest="pretrained",
            action="store_false",
            help="Do not load pretrained weights",
        )
        parser.set_defaults(pretrained=None)


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    """Select the compute device used for a run."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(args: argparse.Namespace, allow_override: bool = True, drop_path: Optional[float] = None) -> torch.nn.Module:
    """Instantiate a model according to CLI flags and defaults."""
    spec = MODEL_SPECS[args.model]
    pretrained = spec.default_pretrained
    if not spec.force_pretrained and allow_override and hasattr(args, "pretrained") and args.pretrained is not None:
        pretrained = args.pretrained
    elif spec.force_pretrained:
        pretrained = spec.default_pretrained

    img_size = args.img_size or spec.default_img_size
    builder_kwargs: Dict[str, Any] = {"drop_path": drop_path if drop_path is not None else getattr(args, "drop_path", 0.1)}  # Extra options forwarded to builder.
    model = spec.builder(
        pretrained=pretrained,
        num_classes=args.num_classes,
        img_size=img_size,
        **builder_kwargs,
    )
    return model


def handle_train(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    model = _build_model(args)
    if args.init_checkpoint:
        # Advanced use-case: warm-start from a custom checkpoint rather
        # than the default torchvision/timm initialisation.
        state = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"] if "model_state" in state else state, strict=False)

    model = model.to(device)

    img_size = args.img_size or MODEL_SPECS[args.model].default_img_size
    train_loader, val_loader = get_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.workers,
        pin_memory=not args.no_pin,
        img_size=img_size,
    )

    optimizer_config = OptimizerConfig(
        name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler_config = None
    if args.scheduler != "none":
        scheduler_config = SchedulerConfig(
            name=args.scheduler,
            milestones=tuple(args.milestones),
            gamma=args.gamma,
            t_max=args.t_max,
        )

    checkpoint_name = args.checkpoint_name or f"{args.model}.pth"
    train_config = TrainConfig(
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        amp=args.amp,
        output_dir=Path(args.output_dir),
        checkpoint_name=checkpoint_name,
        resume_path=Path(args.resume) if args.resume else None,
        metrics_path=Path(args.metrics_path) if args.metrics_path else None,
    )

    if args.seed is not None:
        from pruning_lab.train.train_loop import set_seed

        set_seed(args.seed)

    summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_config=optimizer_config,
        train_config=train_config,
        device=device,
        scheduler_config=scheduler_config,
    )

    print(
        json.dumps(
            {
                "best_accuracy": summary.best_accuracy,
                "best_epoch": summary.best_epoch,
                "best_checkpoint": str(summary.best_checkpoint) if summary.best_checkpoint else None,
                "final_checkpoint": str(summary.final_checkpoint) if summary.final_checkpoint else None,
            },
            indent=2,
        )
    )


def handle_test(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    model = _build_model(args, allow_override=False)

    img_size = args.img_size or MODEL_SPECS[args.model].default_img_size
    _, test_loader = get_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.workers,
        pin_memory=not args.no_pin,
        img_size=img_size,
    )

    result = evaluate_checkpoint(
        model=model,
        checkpoint_path=Path(args.checkpoint),
        data_loader=test_loader,
        device=device,
        strict=args.strict,
    )

    print(json.dumps({"loss": result.loss, "accuracy": result.accuracy}, indent=2))


def handle_prune(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)
    model = _build_model(args, allow_override=False, drop_path=args.drop_path)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        print(f"Warning: Missing keys {missing}, unexpected keys {unexpected}")
    if checkpoint.get("masks"):
        apply_masks(model, checkpoint["masks"])

    img_size = args.img_size or MODEL_SPECS[args.model].default_img_size
    train_loader, val_loader = get_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.workers,
        pin_memory=not args.no_pin,
        img_size=img_size,
    )

    if args.prune_type == "unstructured":
        summary = magnitude_unstructured_prune(
            model,
            amount=args.amount,
            include_bias=args.include_bias,
            include_norm=args.include_norm,
        )
    else:
        summary = structured_channel_prune(model, amount=args.amount)

    apply_masks(model, summary.masks)

    output_checkpoint = args.output_checkpoint
    if not output_checkpoint:
        suffix = "unstructured" if args.prune_type == "unstructured" else "structured"
        output_checkpoint = Path("pruning_lab/models_saved") / f"{args.model}_{suffix}.pth"
    output_checkpoint = Path(output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if args.finetune_epochs > 0:
        optimizer_config = OptimizerConfig(
            name=args.optimizer,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        scheduler_config = None
        if args.scheduler != "none":
            scheduler_config = SchedulerConfig(
                name=args.scheduler,
                milestones=tuple(args.milestones),
                gamma=args.gamma,
                t_max=args.t_max,
            )
        train_config = TrainConfig(
            epochs=args.finetune_epochs,
            amp=args.amp,
            output_dir=output_checkpoint.parent,
            checkpoint_name=output_checkpoint.name,
            eval_interval=args.eval_interval,
        )
        summary_ft = train_model(
            model=model.to(device),
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_config=optimizer_config,
            train_config=train_config,
            device=device,
            scheduler_config=scheduler_config,
            masks={k: v.clone() for k, v in summary.masks.items()},
        )
        final_checkpoint = summary_ft.final_checkpoint or output_checkpoint
        final_masks = summary.masks
    else:
        torch.save(
            {
                "model_state": model.state_dict(),
                "masks": {k: v.cpu() for k, v in summary.masks.items()},
                "prune_type": args.prune_type,
                "amount": args.amount,
            },
            output_checkpoint,
        )
        final_checkpoint = output_checkpoint
        final_masks = summary.masks

    sparsity, per_param = summarize_sparsity(final_masks)
    print(
        json.dumps(
            {
                "checkpoint": str(final_checkpoint),
                "global_sparsity": sparsity,
                "per_parameter_sparsity": per_param,
            },
            indent=2,
        )
    )


def main() -> None:
    args = _parse_args()
    if args.command == "train":
        handle_train(args)
    elif args.command == "test":
        handle_test(args)
    elif args.command == "prune":
        handle_prune(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
