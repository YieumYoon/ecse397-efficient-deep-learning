from pathlib import Path
from typing import Tuple, Optional, Union

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 statistics (computed over training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _build_transforms(train: bool, img_size: int) -> transforms.Compose:
    """Return torchvision transforms for CIFAR-10 with optional resizing."""
    resize = []
    if img_size and img_size != 32:
        resize.append(
            transforms.Resize(
                img_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                *resize,
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            *resize,
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_loaders(
    batch_size: int,
    data_dir: Optional[Union[str, Path]] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    img_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 training and test DataLoaders.

    Args:
        batch_size: Per-iteration batch size.
        data_dir: Directory where CIFAR-10 will be downloaded/cached. If None,
            defaults to the sibling directory of this file ("pruning_lab/data").
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory in DataLoader for faster host->GPU transfer.
        img_size: Target size for images. Values >32 resize CIFAR-10 for larger models.

    Returns:
        (train_loader, test_loader)
    """
    if data_dir is None:
        # Default to this file's directory ("pruning_lab/data")
        data_dir = Path(__file__).resolve().parent
    else:
        data_dir = Path(data_dir)

    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        transform=_build_transforms(train=True, img_size=img_size),
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        transform=_build_transforms(train=False, img_size=img_size),
        download=True,
    )

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent,
    )

    return train_loader, test_loader


__all__ = ["get_loaders"]
