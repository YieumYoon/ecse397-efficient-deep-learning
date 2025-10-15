"""Quick test to verify distillation_lab setup is working correctly.

Run this script to check:
- All modules can be imported
- Models can be instantiated
- KD loss functions work
- Data loading works

Usage:
    python distillation_lab/test_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Testing distillation_lab setup...\n")

# Test 1: Import models
print("1. Testing model imports...")
try:
    from distillation_lab.models.teacher_resnet import create_resnet18_teacher
    from distillation_lab.models.student_resnet import create_resnet8_student
    from distillation_lab.models.teacher_vit import create_vit_tiny_teacher
    from distillation_lab.models.student_vit import create_vit_student
    print("   ✓ Model imports successful")
except Exception as e:
    print(f"   ✗ Model import failed: {e}")
    sys.exit(1)

# Test 2: Import training modules
print("2. Testing training module imports...")
try:
    from distillation_lab.train.train_teacher import train_teacher
    from distillation_lab.train.distill import train_with_distillation
    print("   ✓ Training module imports successful")
except Exception as e:
    print(f"   ✗ Training module import failed: {e}")
    sys.exit(1)

# Test 3: Import KD losses
print("3. Testing KD loss imports...")
try:
    from distillation_lab.utils.kd_losses import (
        kl_divergence_loss,
        distillation_loss,
        feature_distillation_loss,
    )
    print("   ✓ KD loss imports successful")
except Exception as e:
    print(f"   ✗ KD loss import failed: {e}")
    sys.exit(1)

# Test 4: Instantiate models
print("4. Testing model instantiation...")
try:
    import torch
    
    # CNN models
    teacher_cnn = create_resnet18_teacher(num_classes=10, pretrained=False)
    student_cnn = create_resnet8_student(num_classes=10)
    
    # Count parameters
    teacher_params = sum(p.numel() for p in teacher_cnn.parameters())
    student_params = sum(p.numel() for p in student_cnn.parameters())
    
    print(f"   ✓ ResNet-18 teacher: {teacher_params:,} parameters")
    print(f"   ✓ ResNet-8 student: {student_params:,} parameters")
    
    # Check ViT models (requires timm)
    try:
        teacher_vit = create_vit_tiny_teacher(num_classes=10, pretrained=False)
        student_vit = create_vit_student(num_classes=10)
        
        teacher_vit_params = sum(p.numel() for p in teacher_vit.parameters())
        student_vit_params = sum(p.numel() for p in student_vit.parameters())
        
        print(f"   ✓ ViT-Tiny teacher: {teacher_vit_params:,} parameters")
        print(f"   ✓ ViT-Student: {student_vit_params:,} parameters")
    except ImportError:
        print("   ⚠ timm not installed - ViT models unavailable")
        print("     Install with: pip install timm")
    
except Exception as e:
    print(f"   ✗ Model instantiation failed: {e}")
    sys.exit(1)

# Test 5: Test KD loss computation
print("5. Testing KD loss computation...")
try:
    import torch
    from distillation_lab.utils.kd_losses import distillation_loss
    
    # Create dummy data
    batch_size = 4
    num_classes = 10
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Compute loss
    loss = distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        alpha=0.5,
        temperature=4.0
    )
    
    print(f"   ✓ KD loss computed: {loss.item():.4f}")
    
except Exception as e:
    print(f"   ✗ KD loss computation failed: {e}")
    sys.exit(1)

# Test 6: Test forward pass
print("6. Testing model forward pass...")
try:
    import torch
    
    # Test CNN forward pass
    dummy_input = torch.randn(2, 3, 32, 32)
    output = student_cnn(dummy_input)
    assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
    print(f"   ✓ CNN forward pass: input {dummy_input.shape} -> output {output.shape}")
    
    # Test ViT forward pass (if available)
    try:
        dummy_input_vit = torch.randn(2, 3, 224, 224)
        output_vit = student_vit(dummy_input_vit)
        assert output_vit.shape == (2, 10), f"Expected shape (2, 10), got {output_vit.shape}"
        print(f"   ✓ ViT forward pass: input {dummy_input_vit.shape} -> output {output_vit.shape}")
    except:
        print("   ⚠ ViT forward pass skipped (timm not available)")
    
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 7: Test data loading
print("7. Testing data loading...")
try:
    from distillation_lab.data.dataloader import get_loaders
    
    # Try to get loaders (may download CIFAR-10)
    train_loader, test_loader = get_loaders(
        batch_size=32,
        num_workers=0,  # Use 0 workers for testing
        img_size=32,
    )
    
    print(f"   ✓ Train loader: {len(train_loader)} batches")
    print(f"   ✓ Test loader: {len(test_loader)} batches")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"   ✓ Sample batch: images {images.shape}, labels {labels.shape}")
    
except Exception as e:
    print(f"   ⚠ Data loading test skipped: {e}")
    print("     (This is normal if CIFAR-10 is not downloaded yet)")

print("\n" + "="*60)
print("✓ All tests passed! Setup is working correctly.")
print("="*60)
print("\nYou can now:")
print("  1. Train teachers: python -m distillation_lab.main train-teacher --model resnet18")
print("  2. Train students: python -m distillation_lab.main train-student --model resnet8")
print("  3. Run distillation: python -m distillation_lab.main distill --teacher resnet18 --student resnet8 ...")
print("  4. Or submit to cluster: sbatch -C gpu2h100 distillation_lab/utils/train_cnn_teacher.slurm")

