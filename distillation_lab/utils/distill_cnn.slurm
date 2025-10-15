#!/bin/bash
#SBATCH --job-name=distill_cnn
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/logs/distill_cnn_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/logs/distill_cnn_%j.err

# Error handling
set -euo pipefail

# Ensure logs directory exists
mkdir -p $HOME/ecse397-efficient-deep-learning/logs

# CRITICAL: Change to scratch space
WORK_DIR="$TMPDIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "Working directory: $WORK_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Copy project code to scratch
cp -r $HOME/ecse397-efficient-deep-learning/distillation_lab .

# Copy teacher checkpoint to scratch
mkdir -p models_saved
cp $HOME/ecse397-efficient-deep-learning/distillation_lab/models_saved/cnn_teacher.pth models_saved/

# Load PyTorch bundle per cluster guidelines
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Activate virtual environment from home
source $HOME/ecse397-efficient-deep-learning/.venv/bin/activate

echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# Distillation hyperparameters
EPOCHS=${EPOCHS:-200}
LR=${LR:-0.1}
BATCH_SIZE=${BATCH_SIZE:-128}
ALPHA=${ALPHA:-0.5}
TEMPERATURE=${TEMPERATURE:-4.0}

echo "Distillation config: alpha=$ALPHA, temperature=$TEMPERATURE"

# Ensure Python can import the copied package
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

# Run distillation (ResNet-8 from ResNet-18)
python -u -m distillation_lab.main distill \
  --teacher resnet18 \
  --student resnet8 \
  --teacher-checkpoint models_saved/cnn_teacher.pth \
  --alpha "$ALPHA" \
  --temperature "$TEMPERATURE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay 5e-4 \
  --momentum 0.9 \
  --optimizer sgd \
  --scheduler multistep \
  --milestones 100 150 180 \
  --gamma 0.1 \
  --amp \
  --workers 8 \
  --checkpoint-name cnn_student_with_kd.pth \
  --output-dir models_saved

# Copy results back to home directory
echo "Copying results back to home directory..."
mkdir -p $HOME/ecse397-efficient-deep-learning/distillation_lab/models_saved
cp -v models_saved/cnn_student_with_kd.pth $HOME/ecse397-efficient-deep-learning/distillation_lab/models_saved/

# Clean up scratch space
echo "Scratch space will be auto-cleaned by SLURM after job completion"

echo "Job completed successfully!"

