<!-- 45fab8ab-8c30-40cf-ab75-d4e27dd36e98 3e87a305-3ea9-46c6-8f54-b29f32ca5102 -->
# Lab 2: Knowledge Distillation Implementation Plan

## Overview

Implement knowledge distillation to train ResNet-8 and ViT-Student models guided by ResNet-18 and ViT-Tiny teachers, with full cluster automation using SLURM scripts.

## Directory Structure Setup

Create the following structure in `distillation_lab/`:

```
distillation_lab/
├── data/
│   └── dataloader.py          # Reuse from pruning_lab
├── models/
│   ├── teacher_resnet.py      # ResNet-18 wrapper
│   ├── student_resnet.py      # ResNet-8 [1,1,1,1] config
│   ├── teacher_vit.py         # ViT-Tiny wrapper
│   └── student_vit.py         # ViT-6 (6 layers, 192 dim)
├── train/
│   ├── train_teacher.py       # Standard training loop
│   └── distill.py             # KD training with soft targets
├── inference/
│   └── test.py                # Evaluation helpers
├── utils/
│   ├── kd_losses.py           # KD loss functions
│   ├── train_cnn_teacher.slurm
│   ├── train_vit_teacher.slurm
│   ├── train_cnn_student_baseline.slurm
│   ├── train_vit_student_baseline.slurm
│   ├── distill_cnn.slurm
│   └── distill_vit.slurm
├── models_saved/              # Output directory
├── main.py                    # CLI entry point
└── report.json                # Final results
```

## Implementation Tasks

### 1. Data Pipeline

- Copy `pruning_lab/data/dataloader.py` to `distillation_lab/data/`
- Ensure it supports both 32×32 (ResNet) and 224×224 (ViT) image sizes

### 2. Teacher Models

**`models/teacher_resnet.py`:**

- Reuse `pruning_lab/models/resnet18.py` logic (CIFAR-style stem: 3×3 conv, stride=1, no maxpool)
- Create `create_resnet18_teacher()` function
- Prefer CIFAR-10 style training at 32×32; if using ImageNet-pretrained weights, resize inputs to 224, otherwise train CIFAR-style directly

**`models/teacher_vit.py`:**

- Reuse `pruning_lab/models/vit_tiny.py` logic
- Create `create_vit_tiny_teacher()` function
- Use pretrained ViT-Tiny from timm or Hugging Face

### 3. Student Models

**`models/student_resnet.py`:**

- Implement ResNet-8 using PyTorch's `BasicBlock`
- Architecture: `[1, 1, 1, 1]` blocks instead of `[2, 2, 2, 2]`
- Adapt stem for CIFAR-10: 3×3 conv, stride=1, remove maxpool
- Total: ~8 conv layers vs ResNet-18's 16

**`models/student_vit.py`:**

- Implement smaller ViT using timm or custom implementation
- Config: 6 transformer layers (vs 12), 192 embedding dim (vs 384)
- Patch size: 4 or 8 for 32×32 CIFAR-10 images
- Keep num_heads proportional (e.g., 3 heads for 192-dim)

### 4. KD Loss Functions

**`utils/kd_losses.py`:**

Implement three loss components:

```python
def kl_divergence_loss(student_logits, teacher_logits, temperature):
    """KL divergence between soft targets"""
    # Apply temperature scaling
    # Compute log_softmax on student, softmax on teacher
    # Return KL(student||teacher) multiplied by (temperature**2)
    
def distillation_loss(student_logits, teacher_logits, labels, 
                      alpha=0.5, temperature=4.0):
    """Combined KD loss: α·CE + (1-α)·τ²·KL"""
    # Hard target loss: CrossEntropy(student_logits, labels)
    # Soft target loss: (temperature**2) · KL(p_t^τ || p_s^τ)
    # Return weighted combination
    
def feature_distillation_loss(student_features, teacher_features):
    """Optional: FitNets-style feature matching (MSE or cosine)"""
    # Align intermediate representations
    # Return MSE or 1 - cosine_similarity
```

### 5. Training Infrastructure

**`train/train_teacher.py`:**

- Standard supervised training loop
- Use same training logic as `pruning_lab/train/train_loop.py`
- Support both ResNet and ViT with appropriate optimizers/schedulers
- Save best checkpoint based on validation accuracy

**`train/distill.py`:**

- Distillation training loop
- Load frozen teacher model
- For each batch:
  - Get teacher logits (no_grad)
  - Get student logits
  - Compute combined KD loss
  - Backprop through student only
- Track both hard accuracy (on labels) and soft target alignment
- Save best student checkpoint

### 6. CLI Entry Point

**`main.py`:**

Create argparse-based CLI with three modes:

- `train-teacher`: Train teacher models
- `train-student`: Train student without KD (baseline)
- `distill`: Train student with KD

Example usage:

```bash
python -m distillation_lab.main train-teacher --model resnet18 --epochs 300
python -m distillation_lab.main train-student --model resnet8 --epochs 200
python -m distillation_lab.main distill --teacher resnet18 --student resnet8 \
    --teacher-checkpoint models_saved/cnn_teacher.pth --alpha 0.5 --temperature 4.0
```

### 7. SLURM Scripts for Cluster

Based on `pruning_lab/utils/` templates, create 6 SLURM scripts:

**Teacher Training (2 scripts):**

- `utils/train_cnn_teacher.slurm`: Train ResNet-18, 300 epochs, 24h
- `utils/train_vit_teacher.slurm`: Train ViT-Tiny, 100 epochs, 24h

**Student Baseline (2 scripts):**

- `utils/train_cnn_student_baseline.slurm`: Train ResNet-8 without KD, 200 epochs
- `utils/train_vit_student_baseline.slurm`: Train ViT-Student without KD, 150 epochs

**Distillation (2 scripts):**

- `utils/distill_cnn.slurm`: Distill ResNet-8 from ResNet-18, 200 epochs, α=0.5, τ=4.0
- `utils/distill_vit.slurm`: Distill ViT-Student from ViT-Tiny, 150 epochs

All scripts must:

- Use `$TMPDIR` for scratch space
- Copy code/checkpoints to scratch
- Include SLURM resource flags: `--partition=markov_gpu`, `--gres=gpu:1` (add `-C <gpu_type>` if needed)
- Load `Python/3.11.3-GCCcore-12.3.0` and `CUDA/12.1.1` modules
- Activate home virtual environment (e.g., `source $HOME/.venvs/edl/bin/activate`); do NOT copy the venv to scratch
- Copy results back to `models_saved/`
- Use best GPU selection via `submit_best_gpu.sh`

### 8. Report Generation

**Create `generate_report.py` or add to `main.py`:**

- Load all 6 checkpoints
- Evaluate on CIFAR-10 test set
- Generate `report.json`:
```json
{
    "cnn": {
        "teacher_accuracy": 0.912,
        "student_accuracy_without_kd": 0.845,
        "student_accuracy_with_kd": 0.872
    },
    "vit": {
        "teacher_accuracy": 0.927,
        "student_accuracy_without_kd": 0.812,
        "student_accuracy_with_kd": 0.854
    }
}
```


## Key Files to Reference

From `pruning_lab/`:

- `data/dataloader.py` - CIFAR-10 loading with transforms
- `models/resnet18.py` - ResNet architecture helpers
- `models/vit_tiny.py` - ViT implementation
- `train/train_loop.py` - Training loop structure
- `inference/test.py` - Evaluation utilities
- `main.py` - CLI structure with argparse
- `utils/train_*.slurm` - SLURM script templates

From cluster docs:

- Use `$TMPDIR` for scratch space (auto-managed by SLURM)
- Do NOT copy the virtual environment to scratch; activate from `$HOME` after loading modules (e.g., `source $HOME/.venvs/edl/bin/activate`)
- Load `Python/3.11.3-GCCcore-12.3.0` and `CUDA/12.1.1` modules
- Submit via `bash pruning_lab/utils/submit_best_gpu.sh` for optimal GPU

## Training Workflow

1. Train teachers (2 jobs, ~24h each):
   ```bash
   bash pruning_lab/utils/submit_best_gpu.sh distillation_lab/utils/train_cnn_teacher.slurm
   bash pruning_lab/utils/submit_best_gpu.sh distillation_lab/utils/train_vit_teacher.slurm
   ```

2. Train student baselines (2 jobs, ~12-16h each):
   ```bash
   bash pruning_lab/utils/submit_best_gpu.sh distillation_lab/utils/train_cnn_student_baseline.slurm
   bash pruning_lab/utils/submit_best_gpu.sh distillation_lab/utils/train_vit_student_baseline.slurm
   ```

3. Distill students (2 jobs, ~12-16h each):
   ```bash
   bash pruning_lab/utils/submit_best_gpu.sh distillation_lab/utils/distill_cnn.slurm
   bash pruning_lab/utils/submit_best_gpu.sh distillation_lab/utils/distill_vit.slurm
   ```

4. Generate report:
   ```bash
   python -m distillation_lab.main report --models-dir models_saved/
   ```


## Hyperparameters

**Teachers:**

- ResNet-18: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4, scheduler=multistep [150,225,275]
- ViT-Tiny: AdamW, lr=0.001, weight_decay=0.05, scheduler=cosine

**Students (baseline):**

- ResNet-8: Same as ResNet-18 but 200 epochs, scheduler milestones [100,150,180]
- ViT-Student: Same as ViT-Tiny but 150 epochs

**Distillation:**

- α = 0.5 (balance hard/soft targets)
- τ = 4.0 (temperature)
- Same optimizers as baseline students
- Optional: Add `--feature-distill` flag for FitNets-style feature matching

## Expected Outcomes

Based on typical KD results:

- CNN teacher (ResNet-18): ~91-93% accuracy
- CNN student baseline (ResNet-8): ~84-87% accuracy
- CNN student with KD: ~87-89% accuracy (2-3% improvement)
- ViT teacher (ViT-Tiny): ~92-94% accuracy
- ViT student baseline: ~81-85% accuracy
- ViT student with KD: ~85-88% accuracy (3-4% improvement)

### To-dos

- [ ] Create directory structure (data/, models/, train/, inference/, utils/, models_saved/)
- [ ] Copy and adapt dataloader.py from pruning_lab
- [ ] Implement teacher_resnet.py and teacher_vit.py (wrappers around existing models)
- [ ] Implement student_resnet.py (ResNet-8 with [1,1,1,1] config)
- [ ] Implement student_vit.py (ViT with 6 layers, 192 embedding dim)
- [ ] Implement utils/kd_losses.py (KL divergence, combined loss, optional feature distillation)
- [ ] Implement train/train_teacher.py (standard supervised training)
- [ ] Implement train/distill.py (KD training loop with soft targets)
- [ ] Copy and adapt inference/test.py from pruning_lab
- [ ] Implement main.py CLI with train-teacher, train-student, distill, and report modes
- [ ] Create SLURM scripts for teacher training (train_cnn_teacher.slurm, train_vit_teacher.slurm)
- [ ] Create SLURM scripts for student baseline training (train_cnn_student_baseline.slurm, train_vit_student_baseline.slurm)
- [ ] Create SLURM scripts for distillation (distill_cnn.slurm, distill_vit.slurm)
- [ ] Implement report generation functionality to create report.json with all accuracies