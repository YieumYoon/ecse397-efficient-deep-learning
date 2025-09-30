"""Utility helpers for building CIFAR-10 data pipelines.

Background:
    Deep-learning projects almost always start with a `DataLoader`.
    PyTorch splits the input pipeline into two parts:

    1. **Dataset** objects that know how to read raw examples.  We use
       ``torchvision.datasets.CIFAR10`` which automatically downloads
       and caches the dataset.
    2. **Transforms** that preprocess each image before it is packaged
       into a mini-batch.  These include normalisation and data
       augmentation (random crops, flips, etc.).

    Dataset specifics:
        CIFAR-10 is a benchmark vision dataset containing 60,000 colour
        images of size 32×32.  The training set has 50,000 examples and
        the test set has 10,000.  Each image belongs to one of 10
        classes (airplane, automobile, bird, cat, deer, dog, frog,
        horse, ship, truck).  Because images are tiny, data augmentation
        is crucial to help the model generalise.

    Training high-accuracy models requires the data pipeline to be
    carefully tuned.  If we skip normalisation or augmentations, the
    network tends to overfit and will not reach the accuracy thresholds
    required before pruning.  By centralising the logic in this module
    we ensure every experiment in the lab uses the same procedure.

How this fits into Lab 1:
    - The convolutional baseline (ResNet-18) expects 32x32 inputs.
    - The transformer baseline (ViT-Tiny) expects up-sampled inputs
      (typically 224x224) because it was originally trained on
      ImageNet-sized images.
    - Both models call ``get_loaders`` so we expose an ``img_size``
      parameter that can transparently resize data for either case.

Learning goal:
    Spend time reading through the transforms and arguments—you should
    be comfortable adjusting data augmentation policies or dataset
    directories whenever you start a new project.
"""

from pathlib import Path  # Handles filesystem paths in an OS-agnostic way.
import os  # Access SLURM/TMPDIR environment variables when running on the cluster.
from typing import Tuple, Optional, Union

from torch.utils.data import DataLoader  # Batches dataset samples for training.
from torchvision import datasets, transforms  # Provides ready-made datasets and image transforms.


# CIFAR-10 channel statistics (mean and standard deviation).
# These were computed over the 50,000 training images and are commonly
# used in literature to normalise the dataset.
# Normalising by these values makes optimisation more stable and
# enables us to reuse pre-trained models that expect inputs roughly in
# the [-1, 1] range.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # Average pixel values per RGB channel.
CIFAR10_STD = (0.2023, 0.1994, 0.2010)  # Standard deviation per channel.


def _build_transforms(train: bool, img_size: int) -> transforms.Compose:
    """Create the torchvision transform stack used by the loaders.

    Args:
        train: Whether we are preparing augmentations for the training
            set. Training pipelines add random perturbations while the
            test pipeline should remain deterministic.
        img_size: Target spatial resolution.  ResNet-18 works natively
            at 32×32, but ViT-Tiny expects larger images (typically
            224×224).  We optionally insert a resize step so both
            models can share one loader implementation.

    Returns:
        A composed transform function that can be passed directly to
        torchvision datasets.
    """

    # Portion that optionally resizes the input if the model requires a
    # larger image (e.g. ViT).  Using bicubic interpolation preserves as
    # much detail as possible when scaling tiny CIFAR images.
    resize = []  # Will hold an optional Resize transform if we need to scale images.
    if img_size and img_size != 32:
        resize.append(
            transforms.Resize(
                img_size,  # Target edge size for both height and width.
                interpolation=transforms.InterpolationMode.BICUBIC,  # High-quality resizing method.
            )
        )

    if train:
        # Training transformations implement the classic CIFAR recipe:
        # random crops and flips encourage robustness and reduce
        # overfitting before we even start pruning experiments.
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),  # Randomly crops with padding to simulate translations.
                transforms.RandomHorizontalFlip(),  # Mirrors images horizontally with 50% probability.
                *resize,  # Optional upscale step for ViT inputs.
                transforms.ToTensor(),  # Converts PIL images to PyTorch tensors (scales to [0,1]).
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # Standardises pixel intensity per channel.
            ]
        )

    # Evaluation pipeline remains deterministic so reported metrics are
    # stable across runs.
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
    """Create the training and test DataLoaders used throughout the lab.

    The lab handout mandates that everyone use the same CIFAR-10
    pipeline so results remain comparable.  This function handles data
    download, augmentation, and efficient loading.  All training,
    pruning, and evaluation scripts call `get_loaders` to obtain
    iterables of `(image, label)` batches.

    Args:
        batch_size: How many images each mini-batch should contain.
        data_dir: Optional override for where CIFAR-10 should be stored.
            When ``None`` we place the data next to this file to keep
            the repository tidy.
        num_workers: Loader worker processes.  Increasing this speeds up
            CPU preprocessing when you train the big models on a GPU.
        pin_memory: Whether to enable CUDA pinned memory, which speeds
            up host→device copies when using a GPU.
        img_size: Desired spatial size.  Leave at 32 for ResNet-18;
            increase to 224 when training ViT-Tiny.

    Returns:
        A tuple containing the training and test ``DataLoader``
        instances.
    """

    if data_dir is None:
        # Prefer job-local scratch space when available (on Markov via SLURM),
        # otherwise fall back to a repository-local directory.
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir:
            # Keep dataset under TMPDIR so downloads and I/O stay on fast scratch.
            data_dir = Path(tmpdir) / "data" / "cifar10"
        else:
            # Repository-local default keeps things self-contained on non-cluster runs.
            data_dir = Path(__file__).resolve().parent
    else:
        data_dir = Path(data_dir)

    # torchvision handles downloading and caching automatically.  The
    # transforms incorporate the augmentations defined above.
    train_dataset = datasets.CIFAR10(
        root=str(data_dir),  # Directory where files will be stored.
        train=True,  # Indicates we want the training split.
        transform=_build_transforms(train=True, img_size=img_size),  # Augmentation pipeline.
        download=True,  # Downloads automatically if files are missing.
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        transform=_build_transforms(train=False, img_size=img_size),
        download=True,
    )

    # We keep worker processes alive across iterations to avoid Python
    # process start-up costs on subsequent epochs.
    persistent = num_workers > 0  # Keep worker processes alive between epochs to reduce overhead.

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Randomises sample order each epoch (improves convergence).
        num_workers=num_workers,  # Separate processes that load and preprocess data.
        pin_memory=pin_memory,  # Allocates page-locked memory for faster GPU transfers.
        drop_last=True,  # Drops last partial batch so all batches are equal-sized.
        persistent_workers=persistent,  # Reuses worker processes across epochs.
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Evaluation must be deterministic to compare runs fairly.
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Keep final smaller batch to evaluate every sample.
        persistent_workers=persistent,
    )

    return train_loader, test_loader


__all__ = ["get_loaders"]
