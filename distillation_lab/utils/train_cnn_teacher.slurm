#!/bin/bash
#SBATCH --job-name=distill_cnn_teacher
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/logs/distill_cnn_teacher_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/logs/distill_cnn_teacher_%j.err

# Error handling
set -euo pipefail

# Ensure logs directory exists
mkdir -p $HOME/ecse397-efficient-deep-learning/logs

# CRITICAL: Change to scratch space (use $TMPDIR on Markov GPU nodes)
WORK_DIR="$TMPDIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Working directory: $WORK_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Copy project code to scratch (not venv - will activate from home)
cp -r $HOME/ecse397-efficient-deep-learning/distillation_lab .

# Load PyTorch bundle per cluster guidelines
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Activate virtual environment from home directory
source $HOME/ecse397-efficient-deep-learning/.venv/bin/activate

echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# Configurable via environment variables
EPOCHS=${EPOCHS:-300}
LR=${LR:-0.1}
BATCH_SIZE=${BATCH_SIZE:-128}
WEIGHT_DECAY=${WEIGHT_DECAY:-5e-4}
MOMENTUM=${MOMENTUM:-0.9}

# Create local output directory
mkdir -p models_saved

# Ensure Python can import the copied package
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# Run teacher training (ResNet-18)
python -u -m distillation_lab.main train-teacher \
  --model resnet18 \
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
  --checkpoint-name cnn_teacher.pth \
  --output-dir models_saved

# Copy results back to home directory
echo "Copying results back to home directory..."
mkdir -p $HOME/ecse397-efficient-deep-learning/distillation_lab/models_saved
cp -v models_saved/cnn_teacher.pth $HOME/ecse397-efficient-deep-learning/distillation_lab/models_saved/

# Clean up scratch space (TMPDIR is auto-cleaned by SLURM)
echo "Scratch space will be auto-cleaned by SLURM after job completion"

echo "Job completed successfully!"

