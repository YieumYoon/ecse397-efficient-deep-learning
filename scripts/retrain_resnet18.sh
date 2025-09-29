#!/bin/bash
#SBATCH --job-name=resnet18_retrain
#SBATCH --partition=markov_gpu
#SBATCH --nodelist=classt23
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/resnet18_retrain_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/resnet18_retrain_%j.err

echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

# Navigate to project directory
cd /home/jxl2244/ecse397-efficient-deep-learning

# Setup modern Python environment using HPC modules
source scripts/activate_python.sh

# Show GPU info
nvidia-smi

# Show Python version
echo ""
echo "Using Python version:"
python3 --version
echo ""

# Train ResNet-18 with improved hyperparameters to achieve ≥90% accuracy
echo ""
echo "Training ResNet-18 with enhanced hyperparameters..."
echo ""

python -m pruning_lab.main train \
  --model resnet18 \
  --pretrained \
  --epochs 300 \
  --batch-size 128 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --momentum 0.9 \
  --optimizer sgd \
  --scheduler multistep \
  --milestones 150 225 275 \
  --gamma 0.1 \
  --amp \
  --workers 8 \
  --checkpoint-name cnn_before_pruning.pth \
  --output-dir pruning_lab/models_saved \
  --seed 42

echo ""
echo "=================================================="
echo "Training completed at: $(date)"
echo "=================================================="

# Test the trained model
echo ""
echo "Testing trained ResNet-18..."
python -m pruning_lab.main test \
  --model resnet18 \
  --checkpoint pruning_lab/models_saved/cnn_before_pruning.pth \
  --batch-size 256

echo ""
echo "Job finished at: $(date)"
