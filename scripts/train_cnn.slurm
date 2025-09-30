#!/bin/bash
#SBATCH --job-name=train_cnn
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/logs/train_cnn_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/logs/train_cnn_%j.err

set -euo pipefail

cd /home/jxl2244/ecse397-efficient-deep-learning

# Load environment (PyTorch bundle module, user pip, versions)
source scripts/env.sh

# Configurable via environment variables
EPOCHS=${EPOCHS:-300}
LR=${LR:-0.1}
BATCH_SIZE=${BATCH_SIZE:-128}
WEIGHT_DECAY=${WEIGHT_DECAY:-5e-4}
MOMENTUM=${MOMENTUM:-0.9}

python -u -m pruning_lab.main train \
  --model resnet18 \
  --pretrained \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --momentum "$MOMENTUM" \
  --optimizer sgd \
  --scheduler multistep \
  --milestones 150 225 275 \
  --gamma 0.1 \
  --amp \
  --workers 8 \
  --checkpoint-name cnn_before_pruning.pth \
  --output-dir pruning_lab/models_saved \
  --seed 42


