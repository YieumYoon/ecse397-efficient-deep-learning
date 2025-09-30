#!/bin/bash
#SBATCH --job-name=prune_cnn
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/logs/prune_cnn_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/logs/prune_cnn_%j.err

set -euo pipefail

cd /home/jxl2244/ecse397-efficient-deep-learning

source scripts/env.sh

# PRUNE_TYPE: unstructured | structured
PRUNE_TYPE=${PRUNE_TYPE:-unstructured}
AMOUNT=${AMOUNT:-0.7}

python -u -m pruning_lab.main prune \
  --model resnet18 \
  --checkpoint pruning_lab/models_saved/cnn_before_pruning.pth \
  --prune-type "$PRUNE_TYPE" \
  --amount "$AMOUNT" \
  --finetune-epochs 50 \
  --batch-size 128 \
  --lr 0.01 \
  --weight-decay 5e-4 \
  --optimizer sgd \
  --scheduler cosine \
  --t-max 50 \
  --amp \
  --workers 8 \
  --output-checkpoint pruning_lab/models_saved/cnn_after_${PRUNE_TYPE}_pruning.pth


