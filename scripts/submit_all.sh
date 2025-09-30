#!/bin/bash
set -euo pipefail

cd /home/jxl2244/ecse397-efficient-deep-learning

echo "Submitting all jobs for Lab 1-2..."

# Ensure logs directory exists
mkdir -p logs

# Train ResNet-18
JOB1=$(sbatch --parsable scripts/train_cnn.slurm)
echo "✓ CNN training: Job $JOB1"

# Train ViT pretrained (independent)
JOB2=$(sbatch --parsable --export=MODEL_TYPE=pretrained scripts/train_vit.slurm)
echo "✓ ViT training: Job $JOB2"

# Prune CNN unstructured (depends on training)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 --export=PRUNE_TYPE=unstructured scripts/prune_cnn.slurm)
echo "✓ CNN unstructured pruning: Job $JOB3 (depends on $JOB1)"

# Prune CNN structured (depends on training)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB1 --export=PRUNE_TYPE=structured scripts/prune_cnn.slurm)
echo "✓ CNN structured pruning: Job $JOB4 (depends on $JOB1)"

# Prune ViT unstructured (depends on training)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB2 --export=PRUNE_TYPE=unstructured scripts/prune_vit.slurm)
echo "✓ ViT unstructured pruning: Job $JOB5 (depends on $JOB2)"

# Prune ViT structured (depends on training)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB2 --export=PRUNE_TYPE=structured scripts/prune_vit.slurm)
echo "✓ ViT structured pruning: Job $JOB6 (depends on $JOB2)"

echo ""
echo "All jobs submitted!"
echo "Monitor: squeue -u $USER"
echo "After completion: python scripts/update_report.py"


