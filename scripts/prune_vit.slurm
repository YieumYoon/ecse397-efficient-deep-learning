#!/bin/bash
#SBATCH --job-name=prune_vit
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/logs/prune_vit_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/logs/prune_vit_%j.err

# Error handling
set -euo pipefail

# CRITICAL: Change to scratch space (use $TMPDIR on Markov GPU nodes)
WORK_DIR="$TMPDIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Working directory: $WORK_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Copy project code to scratch
cp -r $HOME/ecse397-efficient-deep-learning/pruning_lab .
cp -r $HOME/ecse397-efficient-deep-learning/scripts .

# Copy checkpoint from home to scratch
mkdir -p models_saved
cp $HOME/ecse397-efficient-deep-learning/pruning_lab/models_saved/vit_before_pruning.pth models_saved/

# Load environment
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
export PATH=$HOME/.local/bin:$PATH

echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"

# PRUNE_TYPE: unstructured | structured
PRUNE_TYPE=${PRUNE_TYPE:-unstructured}
AMOUNT=${AMOUNT:-0.7}

# Run pruning
python -u -m pruning_lab.main prune \
  --model vit_tiny_pretrained \
  --img-size 224 \
  --checkpoint models_saved/vit_before_pruning.pth \
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
  --output-checkpoint models_saved/vit_after_${PRUNE_TYPE}_pruning.pth

# Copy results back to home directory
echo "Copying results back to home directory..."
mkdir -p $HOME/ecse397-efficient-deep-learning/pruning_lab/models_saved
cp -v models_saved/vit_after_${PRUNE_TYPE}_pruning.pth $HOME/ecse397-efficient-deep-learning/pruning_lab/models_saved/

# Clean up scratch space (TMPDIR is auto-cleaned by SLURM)
echo "Scratch space will be auto-cleaned by SLURM after job completion"

echo "Job completed successfully!"

