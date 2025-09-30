#!/bin/bash
#SBATCH --job-name=prune_vit
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/logs/prune_vit_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/logs/prune_vit_%j.err

set -euo pipefail

cd /home/jxl2244/ecse397-efficient-deep-learning

source scripts/env.sh

# PRUNE_TYPE: unstructured | structured
PRUNE_TYPE=${PRUNE_TYPE:-unstructured}
AMOUNT=${AMOUNT:-0.7}

python -u -m pruning_lab.main prune \
  --model vit_tiny_pretrained \
  --img-size 224 \
  --checkpoint pruning_lab/models_saved/vit_before_pruning.pth \
  --prune-type "$PRUNE_TYPE" \
  --amount "$AMOUNT" \
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
  --output-checkpoint pruning_lab/models_saved/vit_after_${PRUNE_TYPE}_pruning.pth


