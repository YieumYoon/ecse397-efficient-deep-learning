#!/bin/bash
#SBATCH --job-name=cnn_unstructured_prune
#SBATCH --partition=markov_gpu
#SBATCH --nodelist=classt24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/cnn_unstructured_prune_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/cnn_unstructured_prune_%j.err

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

# Prune ResNet-18 with 70% unstructured pruning
echo ""
echo "Applying unstructured pruning to ResNet-18 (70% sparsity)..."
echo ""

python -m pruning_lab.main prune \
  --model resnet18 \
  --checkpoint pruning_lab/models_saved/cnn_before_pruning.pth \
  --prune-type unstructured \
  --amount 0.70 \
  --finetune-epochs 50 \
  --batch-size 128 \
  --lr 0.01 \
  --weight-decay 5e-4 \
  --optimizer sgd \
  --scheduler cosine \
  --t-max 50 \
  --amp \
  --workers 8 \
  --output-checkpoint pruning_lab/models_saved/cnn_after_unstructured_pruning.pth

echo ""
echo "=================================================="
echo "Pruning completed at: $(date)"
echo "=================================================="

# Test the pruned model
echo ""
echo "Testing pruned ResNet-18 (unstructured)..."
python -m pruning_lab.main test \
  --model resnet18 \
  --checkpoint pruning_lab/models_saved/cnn_after_unstructured_pruning.pth \
  --batch-size 256

echo ""
echo "Job finished at: $(date)"
