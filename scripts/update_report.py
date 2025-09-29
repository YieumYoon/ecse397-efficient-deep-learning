#!/usr/bin/env python3
"""
Script to regenerate report.json with updated model accuracies and sparsity metrics.
This should be run after all training and pruning jobs complete.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pruning_lab.data.dataloader import get_loaders
from pruning_lab.models.resnet18 import create_resnet18
from pruning_lab.models.vit_tiny import create_vit_tiny
from pruning_lab.train.prune import apply_masks, summarize_sparsity


def evaluate_model(model, checkpoint_path, data_loader, device):
    """Evaluate a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=False)
    
    # Apply masks if present
    if checkpoint.get("masks"):
        apply_masks(model, checkpoint["masks"])
    
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    
    # Get sparsity if masks present
    sparsity = 0.0
    if checkpoint.get("masks"):
        sparsity, _ = summarize_sparsity(checkpoint["masks"])
    
    return accuracy, sparsity


def main():
    print("=" * 80)
    print("UPDATING REPORT.JSON WITH NEW RESULTS")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading CIFAR-10 test data...")
    _, test_loader_32 = get_loaders(batch_size=256, img_size=32, num_workers=4)
    _, test_loader_224 = get_loaders(batch_size=256, img_size=224, num_workers=4)
    
    models_dir = Path("pruning_lab/models_saved")
    report = {
        " initial_accuracies ": {},
        " unstructured_pruning ": {"  cnn ": {}, " vit ": {}},
        " structured_pruning ": {" cnn ": {}, " vit ": {}}
    }
    
    # 1. Evaluate CNN before pruning
    print("\n1. Evaluating ResNet-18 before pruning...")
    cnn_model = create_resnet18(num_classes=10, pretrained=False)
    cnn_before_acc, _ = evaluate_model(
        cnn_model, 
        models_dir / "cnn_before_pruning.pth",
        test_loader_32,
        device
    )
    print(f"   CNN Initial Accuracy: {cnn_before_acc:.4f}")
    report[" initial_accuracies "][" cnn_before_pruning "] = cnn_before_acc
    
    # 2. Evaluate ViT before pruning
    print("\n2. Evaluating ViT-Tiny before pruning...")
    vit_model = create_vit_tiny(num_classes=10, pretrained=True, img_size=224)
    vit_before_acc, _ = evaluate_model(
        vit_model,
        models_dir / "vit_before_pruning.pth",
        test_loader_224,
        device
    )
    print(f"   ViT Initial Accuracy: {vit_before_acc:.4f}")
    report[" initial_accuracies "][" vit_before_pruning "] = vit_before_acc
    
    # 3. Evaluate CNN unstructured pruning
    print("\n3. Evaluating ResNet-18 after unstructured pruning...")
    cnn_model = create_resnet18(num_classes=10, pretrained=False)
    cnn_unstruct_acc, cnn_unstruct_sparsity = evaluate_model(
        cnn_model,
        models_dir / "cnn_after_unstructured_pruning.pth",
        test_loader_32,
        device
    )
    print(f"   CNN Unstructured Accuracy: {cnn_unstruct_acc:.4f}")
    print(f"   CNN Unstructured Sparsity: {cnn_unstruct_sparsity * 100:.2f}%")
    report[" unstructured_pruning "][" cnn "] = {
        " original_accuracy ": cnn_before_acc,
        " pruned_accuracy ": cnn_unstruct_acc,
        " pruning_percentage ": cnn_unstruct_sparsity * 100
    }
    
    # 4. Evaluate ViT unstructured pruning
    print("\n4. Evaluating ViT-Tiny after unstructured pruning...")
    vit_model = create_vit_tiny(num_classes=10, pretrained=True, img_size=224)
    vit_unstruct_acc, vit_unstruct_sparsity = evaluate_model(
        vit_model,
        models_dir / "vit_after_unstructured_pruning.pth",
        test_loader_224,
        device
    )
    print(f"   ViT Unstructured Accuracy: {vit_unstruct_acc:.4f}")
    print(f"   ViT Unstructured Sparsity: {vit_unstruct_sparsity * 100:.2f}%")
    report[" unstructured_pruning "][" vit "] = {
        " original_accuracy ": vit_before_acc,
        " pruned_accuracy ": vit_unstruct_acc,
        " pruning_percentage ": vit_unstruct_sparsity * 100
    }
    
    # 5. Evaluate CNN structured pruning
    print("\n5. Evaluating ResNet-18 after structured pruning...")
    cnn_model = create_resnet18(num_classes=10, pretrained=False)
    cnn_struct_acc, cnn_struct_sparsity = evaluate_model(
        cnn_model,
        models_dir / "cnn_after_structured_pruning.pth",
        test_loader_32,
        device
    )
    print(f"   CNN Structured Accuracy: {cnn_struct_acc:.4f}")
    print(f"   CNN Structured Sparsity: {cnn_struct_sparsity * 100:.2f}%")
    report[" structured_pruning "][" cnn "] = {
        " original_accuracy ": cnn_before_acc,
        " pruned_accuracy ": cnn_struct_acc,
        " pruning_percentage ": cnn_struct_sparsity * 100
    }
    
    # 6. Evaluate ViT structured pruning
    print("\n6. Evaluating ViT-Tiny after structured pruning...")
    vit_model = create_vit_tiny(num_classes=10, pretrained=True, img_size=224)
    vit_struct_acc, vit_struct_sparsity = evaluate_model(
        vit_model,
        models_dir / "vit_after_structured_pruning.pth",
        test_loader_224,
        device
    )
    print(f"   ViT Structured Accuracy: {vit_struct_acc:.4f}")
    print(f"   ViT Structured Sparsity: {vit_struct_sparsity * 100:.2f}%")
    report[" structured_pruning "][" vit "] = {
        " original_accuracy ": vit_before_acc,
        " pruned_accuracy ": vit_struct_acc,
        " pruning_percentage ": vit_struct_sparsity * 100
    }
    
    # Save report
    report_path = Path("pruning_lab/report.json")
    print(f"\n\nSaving report to {report_path}...")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 80)
    print("REPORT SUMMARY")
    print("=" * 80)
    print(f"\nInitial Accuracies:")
    print(f"  CNN:  {cnn_before_acc:.2%} (Target: ≥90%)")
    print(f"  ViT:  {vit_before_acc:.2%} (Target: ≥92%)")
    
    print(f"\nUnstructured Pruning:")
    print(f"  CNN:  Acc={cnn_unstruct_acc:.2%}, Sparsity={cnn_unstruct_sparsity*100:.2f}%")
    print(f"  ViT:  Acc={vit_unstruct_acc:.2%}, Sparsity={vit_unstruct_sparsity*100:.2f}%")
    
    print(f"\nStructured Pruning:")
    print(f"  CNN:  Acc={cnn_struct_acc:.2%}, Sparsity={cnn_struct_sparsity*100:.2f}%")
    print(f"  ViT:  Acc={vit_struct_acc:.2%}, Sparsity={vit_struct_sparsity*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("✓ Report updated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

