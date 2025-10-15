"""Training loop for teacher models.

This module provides standard supervised training for ResNet-18 and ViT-Tiny
teacher models on CIFAR-10. It reuses the training infrastructure from the
pruning lab with appropriate configurations.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = False,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float]:
    """Train model for one epoch.
    
    Args:
        model: Neural network to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        use_amp: Use automatic mixed precision
        grad_clip: Gradient clipping value (None to disable)
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    scaler = amp.GradScaler() if use_amp else None
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on test/validation set.
    
    Args:
        model: Neural network to evaluate
        test_loader: Test/validation data loader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epochs: int,
    device: torch.device,
    output_path: Path,
    use_amp: bool = False,
    grad_clip: Optional[float] = None,
) -> dict:
    """Train teacher model and save best checkpoint.
    
    Args:
        model: Teacher model to train
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        epochs: Number of epochs to train
        device: Device to train on
        output_path: Path to save checkpoint
        use_amp: Use automatic mixed precision
        grad_clip: Gradient clipping value
    
    Returns:
        Training history dict
    """
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    history = []
    
    print(f"Training teacher model for {epochs} epochs")
    print(f"Device: {device}")
    print(f"Output path: {output_path}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, use_amp, grad_clip
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': current_lr,
        })
        
        # Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"New best accuracy: {best_acc:.2f}% - Saving checkpoint")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history,
            }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, output_path)
    
    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    
    return {
        'best_accuracy': best_acc,
        'history': history,
    }

