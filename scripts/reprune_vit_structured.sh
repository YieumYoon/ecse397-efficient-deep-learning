#!/bin/bash
#SBATCH --job-name=vit_structured_prune
#SBATCH --partition=markov_gpu
#SBATCH --nodelist=classt06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/vit_structured_prune_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/vit_structured_prune_%j.err

echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

cd /home/jxl2244/ecse397-efficient-deep-learning

# Setup modern Python environment using HPC modules
source scripts/activate_python.sh

# Show GPU info
nvidia-smi
echo ""
python3 --version
echo ""

# Prune ViT-Tiny with increased structured pruning to achieve ≥25% sparsity
echo ""
echo "Applying structured pruning to ViT-Tiny (target: ≥25% sparsity)..."
echo ""

python -m pruning_lab.main prune \
  --model vit_tiny_pretrained \
  --checkpoint pruning_lab/models_saved/vit_before_pruning.pth \
  --prune-type structured \
  --amount 0.35 \
  --finetune-epochs 50 \
  --batch-size 128 \
  --lr 0.005 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --scheduler cosine \
  --t-max 50 \
  --amp \
  --workers 8 \
  --drop-path 0.1 \
  --output-checkpoint pruning_lab/models_saved/vit_after_structured_pruning.pth

echo ""
echo "=================================================="
echo "Pruning completed at: $(date)"
echo "=================================================="

# Test the pruned model
echo ""
echo "Testing pruned ViT-Tiny (structured)..."
python -m pruning_lab.main test \
  --model vit_tiny_pretrained \
  --checkpoint pruning_lab/models_saved/vit_after_structured_pruning.pth \
  --batch-size 256

echo ""
echo "Job finished at: $(date)"
