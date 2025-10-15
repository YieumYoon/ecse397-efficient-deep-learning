"""Knowledge distillation training loop.

This module implements the training loop for student models learning from
teacher models via knowledge distillation.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from distillation_lab.utils.kd_losses import distillation_loss


def distill_epoch(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 0.5,
    temperature: float = 4.0,
    use_amp: bool = False,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Train student model with knowledge distillation for one epoch.
    
    Args:
        student: Student model to train
        teacher: Teacher model (frozen)
        train_loader: Training data loader
        optimizer: Optimizer for student
        device: Device to train on
        alpha: Weight for hard targets (1-alpha for soft targets)
        temperature: Temperature for distillation
        use_amp: Use automatic mixed precision
        grad_clip: Gradient clipping value
    
    Returns:
        Tuple of (average_loss, hard_accuracy, soft_accuracy)
    """
    student.train()
    teacher.eval()  # Teacher is always in eval mode
    
    total_loss = 0.0
    hard_correct = 0  # Correct predictions on hard targets (labels)
    total = 0
    
    scaler = amp.GradScaler() if use_amp else None
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Distilling")):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with amp.autocast():
                # Get student predictions
                student_logits = student(inputs)
                
                # Get teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                
                # Compute distillation loss
                loss = distillation_loss(
                    student_logits,
                    teacher_logits,
                    targets,
                    alpha=alpha,
                    temperature=temperature
                )
            
            scaler.scale(loss).backward()
            
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Get student predictions
            student_logits = student(inputs)
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            
            # Compute distillation loss
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                targets,
                alpha=alpha,
                temperature=temperature
            )
            
            loss.backward()
            
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            
            optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += targets.size(0)
        hard_correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    hard_accuracy = 100.0 * hard_correct / total
    
    return avg_loss, hard_accuracy


def evaluate_student(
    student: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate student model on test set.
    
    Args:
        student: Student model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    student.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epochs: int,
    device: torch.device,
    output_path: Path,
    alpha: float = 0.5,
    temperature: float = 4.0,
    use_amp: bool = False,
    grad_clip: Optional[float] = None,
) -> dict:
    """Train student model with knowledge distillation.
    
    Args:
        student: Student model to train
        teacher: Teacher model (frozen)
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer for student
        scheduler: Learning rate scheduler (optional)
        epochs: Number of epochs to train
        device: Device to train on
        output_path: Path to save checkpoint
        alpha: Weight for hard targets
        temperature: Temperature for distillation
        use_amp: Use automatic mixed precision
        grad_clip: Gradient clipping value
    
    Returns:
        Training history dict
    """
    best_acc = 0.0
    history = []
    
    print(f"Training student with knowledge distillation for {epochs} epochs")
    print(f"Device: {device}")
    print(f"Alpha: {alpha}, Temperature: {temperature}")
    print(f"Output path: {output_path}")
    
    # Freeze teacher
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train with distillation
        train_loss, train_acc = distill_epoch(
            student, teacher, train_loader, optimizer,
            device, alpha, temperature, use_amp, grad_clip
        )
        
        # Evaluate
        test_loss, test_acc = evaluate_student(student, test_loader, device)
        
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
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'alpha': alpha,
                'temperature': temperature,
                'history': history,
            }
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, output_path)
    
    print(f"\nDistillation complete! Best accuracy: {best_acc:.2f}%")
    
    return {
        'best_accuracy': best_acc,
        'history': history,
    }

