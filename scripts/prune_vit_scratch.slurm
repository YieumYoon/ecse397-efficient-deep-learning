#!/bin/bash
#SBATCH --job-name=prune_vit_scratch
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/logs/prune_vit_scratch_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/logs/prune_vit_scratch_%j.err

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
cp -r $HOME/ecse397-efficient-deep-learning/.venv .

# Load environment - Python 3.11 and CUDA (NOT PyTorch module)
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Activate virtual environment with PyTorch 2.5.1
source .venv/bin/activate

echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# Configuration
MODEL=vit_tiny_scratch
CHECKPOINT_PATH=pruning_lab/models_saved/vit_scratch_before_pruning.pth
PRUNE_TYPE=${PRUNE_TYPE:-unstructured}  # unstructured | structured
AMOUNT=${AMOUNT:-0.70}

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
  echo "Please ensure ViT-scratch training has completed."
  exit 1
fi

echo "Pruning configuration:"
echo "  Model: $MODEL"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Prune type: $PRUNE_TYPE"
echo "  Amount: $AMOUNT"

# Determine output checkpoint name
if [ "$PRUNE_TYPE" = "unstructured" ]; then
  OUTPUT_NAME="vit_scratch_after_unstructured_pruning.pth"
else
  OUTPUT_NAME="vit_scratch_after_structured_pruning.pth"
fi

# Create local output directory
mkdir -p models_saved

# Run pruning with fine-tuning
python -u -m pruning_lab.main prune \
  --model "$MODEL" \
  --checkpoint "$CHECKPOINT_PATH" \
  --prune-type "$PRUNE_TYPE" \
  --amount "$AMOUNT" \
  --finetune-epochs 20 \
  --lr 0.0001 \
  --optimizer adamw \
  --scheduler cosine \
  --t-max 20 \
  --amp \
  --workers 8 \
  --output-checkpoint "models_saved/$OUTPUT_NAME"

# Evaluate the pruned model
echo ""
echo "Evaluating pruned model..."
python -u -m pruning_lab.main test \
  --model "$MODEL" \
  --checkpoint "models_saved/$OUTPUT_NAME" \
  --batch-size 256 \
  --img-size 224

# Copy results back to home directory
echo ""
echo "Copying results back to home directory..."
mkdir -p $HOME/ecse397-efficient-deep-learning/pruning_lab/models_saved
cp -v models_saved/"$OUTPUT_NAME" $HOME/ecse397-efficient-deep-learning/pruning_lab/models_saved/

# Clean up scratch space (TMPDIR is auto-cleaned by SLURM)
echo "Scratch space will be auto-cleaned by SLURM after job completion"

echo ""
echo "Pruning completed successfully!"
echo "Output checkpoint: pruning_lab/models_saved/$OUTPUT_NAME"
