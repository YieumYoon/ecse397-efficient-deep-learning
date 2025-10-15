# Lab 2: Knowledge Distillation

This lab implements knowledge distillation to train compact student models (ResNet-8 and ViT-Student) guided by larger teacher models (ResNet-18 and ViT-Tiny) on CIFAR-10.

## Directory Structure

```
distillation_lab/
├── data/
│   └── dataloader.py          # CIFAR-10 data loading
├── models/
│   ├── teacher_resnet.py      # ResNet-18 teacher
│   ├── student_resnet.py      # ResNet-8 student
│   ├── teacher_vit.py         # ViT-Tiny teacher
│   └── student_vit.py         # ViT-Student (6 layers, 192 dim)
├── train/
│   ├── train_teacher.py       # Standard training loop
│   └── distill.py             # Knowledge distillation training
├── inference/
│   └── test.py                # Evaluation utilities
├── utils/
│   ├── kd_losses.py           # KD loss functions
│   └── *.slurm                # SLURM job scripts
├── models_saved/              # Output checkpoints
├── main.py                    # CLI entry point
└── report.json                # Final results
```

## Quick Start (Local)

### 1. Train Teacher Models

```bash
# Train ResNet-18 teacher
python -m distillation_lab.main train-teacher --model resnet18 --epochs 300

# Train ViT-Tiny teacher
python -m distillation_lab.main train-teacher --model vit_tiny --epochs 100
```

### 2. Train Student Baselines (without KD)

```bash
# Train ResNet-8 baseline
python -m distillation_lab.main train-student --model resnet8 --epochs 200

# Train ViT-Student baseline
python -m distillation_lab.main train-student --model vit_student --epochs 150
```

### 3. Train Students with Distillation

```bash
# Distill ResNet-8 from ResNet-18
python -m distillation_lab.main distill \
  --teacher resnet18 \
  --student resnet8 \
  --teacher-checkpoint distillation_lab/models_saved/cnn_teacher.pth \
  --alpha 0.5 \
  --temperature 4.0 \
  --epochs 200

# Distill ViT-Student from ViT-Tiny
python -m distillation_lab.main distill \
  --teacher vit_tiny \
  --student vit_student \
  --teacher-checkpoint distillation_lab/models_saved/vit_teacher.pth \
  --alpha 0.5 \
  --temperature 4.0 \
  --epochs 150
```

### 4. Generate Report

```bash
python -m distillation_lab.main report --models-dir distillation_lab/models_saved
```

## Cluster Usage


### Complete Workflow

```bash
# 1. Train teachers
sbatch -C gpu2h100 distillation_lab/utils/train_cnn_teacher.slurm
sbatch -C gpu2h100 distillation_lab/utils/train_vit_teacher.slurm

# 2. Wait for teachers to complete, then train student baselines
sbatch -C gpu2h100 distillation_lab/utils/train_cnn_student_baseline.slurm
sbatch -C gpu2h100 distillation_lab/utils/train_vit_student_baseline.slurm

# 3. Train students with distillation
sbatch -C gpu2h100 distillation_lab/utils/distill_cnn.slurm
sbatch -C gpu2h100 distillation_lab/utils/distill_vit.slurm

# 4. Generate report (CPU OK)
module purge && module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
python -m distillation_lab.main report --models-dir distillation_lab/models_saved
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View live logs
tail -f logs/distill_*_<job_id>.out

# Check completed job output
cat logs/distill_cnn_teacher_<job_id>.out
```

## Model Architectures

### Teachers
- **ResNet-18**: 11.2M parameters, pretrained on ImageNet
- **ViT-Tiny**: 5.7M parameters, 12 layers, 192 embedding dim

### Students
- **ResNet-8**: ~2.8M parameters, [1,1,1,1] block configuration
- **ViT-Student**: ~1.4M parameters, 6 layers, 192 embedding dim

## Knowledge Distillation

### Loss Function

```
L_KD = α·L_CE(y, p_s) + (1-α)·τ²·KL(p_t^τ || p_s^τ)
```

Where:
- `α = 0.5`: Weight between hard targets (CE loss) and soft targets (KL divergence)
- `τ = 4.0`: Temperature for softening probability distributions
- `L_CE`: Cross-entropy loss with ground truth labels
- `KL`: KL divergence between teacher and student logits

### Hyperparameters

**CNN (ResNet-8):**
- Optimizer: SGD with momentum 0.9
- Learning rate: 0.1 with MultiStepLR [100, 150, 180]
- Weight decay: 5e-4
- Batch size: 128
- Epochs: 200

**ViT (ViT-Student):**
- Optimizer: AdamW
- Learning rate: 0.001 with CosineAnnealingLR
- Weight decay: 0.05
- Batch size: 128
- Epochs: 150

## Expected Results

Based on typical knowledge distillation outcomes:

| Model | Expected Accuracy |
|-------|------------------|
| ResNet-18 Teacher | 91-93% |
| ResNet-8 Baseline | 84-87% |
| ResNet-8 with KD | 87-89% |
| ViT-Tiny Teacher | 92-94% |
| ViT-Student Baseline | 81-85% |
| ViT-Student with KD | 85-88% |

**Improvement from KD:** 2-4% accuracy gain for student models.

## Troubleshooting

### Import Errors

```bash
# Ensure you're running from the project root
cd /home/jxl2244/ecse397-efficient-deep-learning

# Verify timm is installed
pip install timm
```

### CUDA Out of Memory

- Reduce batch size: `--batch-size 64`
- Disable AMP: Remove `--amp` flag

### Checkpoint Not Found

Ensure teacher training completed before distillation:

```bash
ls -lh distillation_lab/models_saved/cnn_teacher.pth
```

## Submission

1. Ensure all 6 checkpoints are saved in `models_saved/`
2. Generate `report.json`
3. Compress the lab:

```bash
cd /home/jxl2244/ecse397-efficient-deep-learning
zip -r studentID_distillation_lab.zip distillation_lab/
```

4. Submit via Canvas before deadline: October 15, 11:59 pm EST

## References

- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Romero et al., "FitNets: Hints for Thin Deep Nets" (2015)
- [PyTorch KD Example](https://github.com/peterliht/knowledge-distillation-pytorch)

