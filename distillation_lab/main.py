"""Command-line entry point for knowledge distillation experiments.

Usage:
    python -m distillation_lab.main train-teacher --model resnet18 --epochs 300
    python -m distillation_lab.main train-student --model resnet8 --epochs 200
    python -m distillation_lab.main distill --teacher resnet18 --student resnet8 \\
        --teacher-checkpoint models_saved/cnn_teacher.pth --alpha 0.5 --temperature 4.0
    python -m distillation_lab.main report --models-dir models_saved/
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from distillation_lab.data.dataloader import get_loaders
from distillation_lab.models.teacher_resnet import create_resnet18_teacher
from distillation_lab.models.teacher_vit import create_vit_tiny_teacher
from distillation_lab.models.student_resnet import create_resnet8_student
from distillation_lab.models.student_vit import create_vit_student
from distillation_lab.train.train_teacher import train_teacher
from distillation_lab.train.distill import train_with_distillation
from distillation_lab.inference.test import evaluate_checkpoint


# Model registry
TEACHER_MODELS = {
    'resnet18': {
        'builder': create_resnet18_teacher,
        'img_size': 32,
        'pretrained': False,
    },
    'vit_tiny': {
        'builder': create_vit_tiny_teacher,
        'img_size': 224,
        'pretrained': True,
    },
}

STUDENT_MODELS = {
    'resnet8': {
        'builder': create_resnet8_student,
        'img_size': 32,
    },
    'vit_student': {
        'builder': create_vit_student,
        'img_size': 224,
    },
}


def create_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Create optimizer from arguments."""
    if args.optimizer == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def create_scheduler(optimizer: torch.optim.Optimizer, args):
    """Create learning rate scheduler from arguments."""
    if args.scheduler == 'none':
        return None
    elif args.scheduler == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.milestones,
            gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.t_max
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


def cmd_train_teacher(args):
    """Train a teacher model."""
    print(f"Training teacher: {args.model}")
    
    # Get model specification
    if args.model not in TEACHER_MODELS:
        raise ValueError(f"Unknown teacher model: {args.model}. Choose from: {list(TEACHER_MODELS.keys())}")
    
    spec = TEACHER_MODELS[args.model]
    img_size = args.img_size if args.img_size is not None else spec['img_size']
    
    # Create model
    model = spec['builder'](num_classes=10, pretrained=spec['pretrained'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Data loaders
    train_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=not args.no_pin,
        img_size=img_size,
        data_dir=args.data_dir,
    )
    
    # Optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    
    # Output path
    output_dir = Path(args.output_dir)
    if args.checkpoint_name is None:
        checkpoint_name = f"{args.model}_teacher.pth"
    else:
        checkpoint_name = args.checkpoint_name
    output_path = output_dir / checkpoint_name
    
    # Train
    history = train_teacher(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        output_path=output_path,
        use_amp=args.amp,
        grad_clip=args.grad_clip,
    )
    
    print(f"\nCheckpoint saved: {output_path}")
    print(f"Best accuracy: {history['best_accuracy']:.2f}%")


def cmd_train_student(args):
    """Train a student model without distillation (baseline)."""
    print(f"Training student (baseline): {args.model}")
    
    # Get model specification
    if args.model not in STUDENT_MODELS:
        raise ValueError(f"Unknown student model: {args.model}. Choose from: {list(STUDENT_MODELS.keys())}")
    
    spec = STUDENT_MODELS[args.model]
    img_size = args.img_size if args.img_size is not None else spec['img_size']
    
    # Create model
    model = spec['builder'](num_classes=10)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Data loaders
    train_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=not args.no_pin,
        img_size=img_size,
        data_dir=args.data_dir,
    )
    
    # Optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    
    # Output path
    output_dir = Path(args.output_dir)
    if args.checkpoint_name is None:
        checkpoint_name = f"{args.model}_student_no_kd.pth"
    else:
        checkpoint_name = args.checkpoint_name
    output_path = output_dir / checkpoint_name
    
    # Train
    history = train_teacher(  # Use same training function as teacher
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        output_path=output_path,
        use_amp=args.amp,
        grad_clip=args.grad_clip,
    )
    
    print(f"\nCheckpoint saved: {output_path}")
    print(f"Best accuracy: {history['best_accuracy']:.2f}%")


def cmd_distill(args):
    """Train student with knowledge distillation."""
    print(f"Distilling: {args.student} from {args.teacher}")
    
    # Validate models
    if args.teacher not in TEACHER_MODELS:
        raise ValueError(f"Unknown teacher: {args.teacher}")
    if args.student not in STUDENT_MODELS:
        raise ValueError(f"Unknown student: {args.student}")
    
    teacher_spec = TEACHER_MODELS[args.teacher]
    student_spec = STUDENT_MODELS[args.student]
    
    # Get image size (use student's preference)
    img_size = args.img_size if args.img_size is not None else student_spec['img_size']
    
    # Create models
    teacher = teacher_spec['builder'](num_classes=10, pretrained=teacher_spec['pretrained'])
    student = student_spec['builder'](num_classes=10)
    
    # Load teacher checkpoint
    print(f"Loading teacher checkpoint: {args.teacher_checkpoint}")
    checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state_dict'])
    else:
        teacher.load_state_dict(checkpoint)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = teacher.to(device)
    student = student.to(device)
    
    print(f"Teacher: {args.teacher} ({sum(p.numel() for p in teacher.parameters()):,} params)")
    print(f"Student: {args.student} ({sum(p.numel() for p in student.parameters()):,} params)")
    print(f"Device: {device}")
    
    # Data loaders
    train_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=not args.no_pin,
        img_size=img_size,
        data_dir=args.data_dir,
    )
    
    # Optimizer and scheduler
    optimizer = create_optimizer(student, args)
    scheduler = create_scheduler(optimizer, args)
    
    # Output path
    output_dir = Path(args.output_dir)
    if args.checkpoint_name is None:
        checkpoint_name = f"{args.student}_student_with_kd.pth"
    else:
        checkpoint_name = args.checkpoint_name
    output_path = output_dir / checkpoint_name
    
    # Train with distillation
    history = train_with_distillation(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        output_path=output_path,
        alpha=args.alpha,
        temperature=args.temperature,
        use_amp=args.amp,
        grad_clip=args.grad_clip,
    )
    
    print(f"\nCheckpoint saved: {output_path}")
    print(f"Best accuracy: {history['best_accuracy']:.2f}%")


def cmd_report(args):
    """Generate report.json from all checkpoints."""
    print("Generating report from checkpoints...")
    
    models_dir = Path(args.models_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Expected checkpoints
    checkpoints = {
        'cnn_teacher': 'cnn_teacher.pth',
        'cnn_student_no_kd': 'cnn_student_no_kd.pth',
        'cnn_student_with_kd': 'cnn_student_with_kd.pth',
        'vit_teacher': 'vit_teacher.pth',
        'vit_student_no_kd': 'vit_student_no_kd.pth',
        'vit_student_with_kd': 'vit_student_with_kd.pth',
    }
    
    # Model configurations
    model_configs = {
        'cnn_teacher': ('resnet18', TEACHER_MODELS['resnet18'], 32),
        'cnn_student_no_kd': ('resnet8', STUDENT_MODELS['resnet8'], 32),
        'cnn_student_with_kd': ('resnet8', STUDENT_MODELS['resnet8'], 32),
        'vit_teacher': ('vit_tiny', TEACHER_MODELS['vit_tiny'], 224),
        'vit_student_no_kd': ('vit_student', STUDENT_MODELS['vit_student'], 224),
        'vit_student_with_kd': ('vit_student', STUDENT_MODELS['vit_student'], 224),
    }
    
    results = {}
    
    for key, ckpt_name in checkpoints.items():
        ckpt_path = models_dir / ckpt_name
        
        if not ckpt_path.exists():
            print(f"Warning: {ckpt_path} not found, skipping...")
            continue
        
        print(f"Evaluating {key}...")
        
        # Get model config
        model_name, spec, img_size = model_configs[key]
        
        # Create model
        if 'teacher' in key:
            model = spec['builder'](num_classes=10, pretrained=False)
        else:
            model = spec['builder'](num_classes=10)
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            accuracy = checkpoint.get('best_acc', None)
        else:
            model.load_state_dict(checkpoint)
            accuracy = None
        
        model = model.to(device)
        
        # Evaluate if accuracy not in checkpoint
        if accuracy is None:
            _, test_loader = get_loaders(
                batch_size=256,
                num_workers=4,
                img_size=img_size,
            )
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            accuracy = 100.0 * correct / total
        
        results[key] = accuracy / 100.0  # Convert to decimal
        print(f"  Accuracy: {accuracy:.2f}%")
    
    # Generate report structure
    report = {
        'cnn': {
            'teacher_accuracy': results.get('cnn_teacher', 0.0),
            'student_accuracy_without_kd': results.get('cnn_student_no_kd', 0.0),
            'student_accuracy_with_kd': results.get('cnn_student_with_kd', 0.0),
        },
        'vit': {
            'teacher_accuracy': results.get('vit_teacher', 0.0),
            'student_accuracy_without_kd': results.get('vit_student_no_kd', 0.0),
            'student_accuracy_with_kd': results.get('vit_student_with_kd', 0.0),
        },
    }
    
    # Save report
    output_path = Path(args.output) if args.output else Path('distillation_lab/report.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nReport saved: {output_path}")
    print(json.dumps(report, indent=2))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation Lab CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train teacher command
    train_teacher_parser = subparsers.add_parser('train-teacher', help='Train teacher model')
    train_teacher_parser.add_argument('--model', type=str, required=True, choices=list(TEACHER_MODELS.keys()))
    train_teacher_parser.add_argument('--epochs', type=int, default=300)
    train_teacher_parser.add_argument('--batch-size', type=int, default=128)
    train_teacher_parser.add_argument('--lr', type=float, default=0.1)
    train_teacher_parser.add_argument('--weight-decay', type=float, default=5e-4)
    train_teacher_parser.add_argument('--momentum', type=float, default=0.9)
    train_teacher_parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    train_teacher_parser.add_argument('--scheduler', type=str, default='multistep', choices=['none', 'multistep', 'cosine'])
    train_teacher_parser.add_argument('--milestones', type=int, nargs='*', default=[150, 225, 275])
    train_teacher_parser.add_argument('--gamma', type=float, default=0.1)
    train_teacher_parser.add_argument('--t-max', type=int, default=300)
    train_teacher_parser.add_argument('--grad-clip', type=float, default=None)
    train_teacher_parser.add_argument('--amp', action='store_true')
    train_teacher_parser.add_argument('--img-size', type=int, default=None)
    train_teacher_parser.add_argument('--data-dir', type=str, default=None)
    train_teacher_parser.add_argument('--workers', type=int, default=4)
    train_teacher_parser.add_argument('--no-pin', action='store_true')
    train_teacher_parser.add_argument('--output-dir', type=str, default='distillation_lab/models_saved')
    train_teacher_parser.add_argument('--checkpoint-name', type=str, default=None)
    
    # Train student command
    train_student_parser = subparsers.add_parser('train-student', help='Train student model without KD')
    train_student_parser.add_argument('--model', type=str, required=True, choices=list(STUDENT_MODELS.keys()))
    train_student_parser.add_argument('--epochs', type=int, default=200)
    train_student_parser.add_argument('--batch-size', type=int, default=128)
    train_student_parser.add_argument('--lr', type=float, default=0.1)
    train_student_parser.add_argument('--weight-decay', type=float, default=5e-4)
    train_student_parser.add_argument('--momentum', type=float, default=0.9)
    train_student_parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    train_student_parser.add_argument('--scheduler', type=str, default='multistep', choices=['none', 'multistep', 'cosine'])
    train_student_parser.add_argument('--milestones', type=int, nargs='*', default=[100, 150, 180])
    train_student_parser.add_argument('--gamma', type=float, default=0.1)
    train_student_parser.add_argument('--t-max', type=int, default=200)
    train_student_parser.add_argument('--grad-clip', type=float, default=None)
    train_student_parser.add_argument('--amp', action='store_true')
    train_student_parser.add_argument('--img-size', type=int, default=None)
    train_student_parser.add_argument('--data-dir', type=str, default=None)
    train_student_parser.add_argument('--workers', type=int, default=4)
    train_student_parser.add_argument('--no-pin', action='store_true')
    train_student_parser.add_argument('--output-dir', type=str, default='distillation_lab/models_saved')
    train_student_parser.add_argument('--checkpoint-name', type=str, default=None)
    
    # Distill command
    distill_parser = subparsers.add_parser('distill', help='Train student with distillation')
    distill_parser.add_argument('--teacher', type=str, required=True, choices=list(TEACHER_MODELS.keys()))
    distill_parser.add_argument('--student', type=str, required=True, choices=list(STUDENT_MODELS.keys()))
    distill_parser.add_argument('--teacher-checkpoint', type=str, required=True)
    distill_parser.add_argument('--alpha', type=float, default=0.5, help='Weight for hard targets')
    distill_parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for distillation')
    distill_parser.add_argument('--epochs', type=int, default=200)
    distill_parser.add_argument('--batch-size', type=int, default=128)
    distill_parser.add_argument('--lr', type=float, default=0.1)
    distill_parser.add_argument('--weight-decay', type=float, default=5e-4)
    distill_parser.add_argument('--momentum', type=float, default=0.9)
    distill_parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
    distill_parser.add_argument('--scheduler', type=str, default='multistep', choices=['none', 'multistep', 'cosine'])
    distill_parser.add_argument('--milestones', type=int, nargs='*', default=[100, 150, 180])
    distill_parser.add_argument('--gamma', type=float, default=0.1)
    distill_parser.add_argument('--t-max', type=int, default=200)
    distill_parser.add_argument('--grad-clip', type=float, default=None)
    distill_parser.add_argument('--amp', action='store_true')
    distill_parser.add_argument('--img-size', type=int, default=None)
    distill_parser.add_argument('--data-dir', type=str, default=None)
    distill_parser.add_argument('--workers', type=int, default=4)
    distill_parser.add_argument('--no-pin', action='store_true')
    distill_parser.add_argument('--output-dir', type=str, default='distillation_lab/models_saved')
    distill_parser.add_argument('--checkpoint-name', type=str, default=None)
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate report.json')
    report_parser.add_argument('--models-dir', type=str, default='distillation_lab/models_saved')
    report_parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    # Route to appropriate command
    if args.command == 'train-teacher':
        cmd_train_teacher(args)
    elif args.command == 'train-student':
        cmd_train_student(args)
    elif args.command == 'distill':
        cmd_distill(args)
    elif args.command == 'report':
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

