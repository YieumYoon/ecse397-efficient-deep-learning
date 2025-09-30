#!/bin/bash
#SBATCH --job-name=train_vit
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/logs/train_vit_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/logs/train_vit_%j.err

set -euo pipefail

cd /home/jxl2244/ecse397-efficient-deep-learning

source scripts/env.sh

# MODEL_TYPE: pretrained | scratch
MODEL_TYPE=${MODEL_TYPE:-pretrained}
EPOCHS=${EPOCHS:-100}
LR=${LR:-0.001}

if [ "$MODEL_TYPE" = "pretrained" ]; then
  MODEL=vit_tiny_pretrained
  CKPT_NAME=vit_before_pruning.pth
else
  MODEL=vit_tiny_scratch
  CKPT_NAME=vit_scratch_before_pruning.pth
fi

python -u -m pruning_lab.main train \
  --model "$MODEL" \
  --img-size 224 \
  --epochs "$EPOCHS" \
  --batch-size 128 \
  --optimizer adamw \
  --lr "$LR" \
  --scheduler cosine \
  --t-max "$EPOCHS" \
  --amp \
  --workers 8 \
  --checkpoint-name "$CKPT_NAME" \
  --output-dir pruning_lab/models_saved


