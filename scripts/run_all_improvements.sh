#!/bin/bash
# Master script to run all improvements in sequence

echo "=================================================="
echo "RUNNING ALL LAB IMPROVEMENTS"
echo "=================================================="
echo ""
echo "This script will:"
echo "  1. Retrain ResNet-18 for ≥90% accuracy"
echo "  2. Re-prune CNN with updated code (unstructured)"
echo "  3. Re-prune ViT with updated code (unstructured)"
echo "  4. Re-prune ViT with higher structured pruning"
echo "  5. Regenerate report.json with all results"
echo ""
echo "GPU Allocation:"
echo "  - classt23: ResNet-18 training"
echo "  - classt24: CNN unstructured pruning"
echo "  - classt25: ViT unstructured pruning"
echo "  - classt06: ViT structured pruning"
echo ""
echo "=================================================="

cd /home/jxl2244/ecse397-efficient-deep-learning

# Step 1: Submit ResNet-18 retraining job
echo ""
echo "Step 1: Submitting ResNet-18 retraining job..."
JOB1=$(sbatch --parsable scripts/retrain_resnet18.sh)
echo "  Job ID: $JOB1 (classt23)"

# Step 2: Submit pruning jobs that depend on training completion
echo ""
echo "Step 2: Submitting CNN unstructured pruning (depends on Job $JOB1)..."
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/reprune_cnn_unstructured.sh)
echo "  Job ID: $JOB2 (classt24)"

# Step 3: Submit ViT unstructured pruning (independent - uses existing checkpoint)
echo ""
echo "Step 3: Submitting ViT unstructured pruning (independent)..."
JOB3=$(sbatch --parsable scripts/reprune_vit_unstructured.sh)
echo "  Job ID: $JOB3 (classt25)"

# Step 4: Submit ViT structured pruning (independent - uses existing checkpoint)
echo ""
echo "Step 4: Submitting ViT structured pruning (independent)..."
JOB4=$(sbatch --parsable scripts/reprune_vit_structured.sh)
echo "  Job ID: $JOB4 (classt06)"

echo ""
echo "=================================================="
echo "All jobs submitted!"
echo "=================================================="
echo ""
echo "Job Dependencies:"
echo "  $JOB1: ResNet-18 training (classt23)"
echo "  $JOB2: CNN unstructured pruning (depends on $JOB1) (classt24)"
echo "  $JOB3: ViT unstructured pruning (classt25)"
echo "  $JOB4: ViT structured pruning (classt06)"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo ""
echo "Once all jobs complete, run:"
echo "  python scripts/update_report.py"
echo ""
echo "Expected completion time:"
echo "  - ResNet training: ~12-18 hours"
echo "  - Pruning jobs: ~2-4 hours each"
echo ""
echo "=================================================="

# Save job IDs to a file for later reference
cat > /home/jxl2244/ecse397-efficient-deep-learning/job_ids.txt << EOF
RESNET_TRAINING=$JOB1
CNN_UNSTRUCTURED=$JOB2
VIT_UNSTRUCTURED=$JOB3
VIT_STRUCTURED=$JOB4
EOF

echo ""
echo "Job IDs saved to job_ids.txt"
echo ""

