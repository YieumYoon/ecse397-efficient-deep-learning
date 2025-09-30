#!/bin/bash
# Wrapper script to automatically select best GPU and submit job
# Usage: bash submit_best_gpu.sh <script.slurm> [optional SLURM args]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script.slurm> [optional SLURM args]"
    echo ""
    echo "Examples:"
    echo "  $0 scripts_corrected/train_cnn.slurm"
    echo "  $0 scripts_corrected/train_vit.slurm --time=12:00:00"
    echo ""
    exit 1
fi

SCRIPT=$1
shift  # Remove first argument, keep the rest

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script not found: $SCRIPT"
    exit 1
fi

# Get best available GPU
echo "Checking GPU availability..."
GPU_TYPE=$(bash scripts/select_best_gpu.sh 2>&1)

echo ""
echo "Selected GPU type: $GPU_TYPE"
echo ""

# Show current GPU status
echo "Current GPU status:"
si | grep markov_gpu | head -10
echo ""

# Submit job with selected GPU constraint
echo "Submitting job: $SCRIPT"
echo "GPU constraint: -C $GPU_TYPE"
echo "Additional args: $@"
echo ""

# Extract GPU type from output (last line)
GPU_FEATURE=$(echo "$GPU_TYPE" | tail -1)

# Submit the job
sbatch -C "$GPU_FEATURE" "$@" "$SCRIPT"

echo ""
echo "Job submitted! Check status with: squeue -u \$USER"
