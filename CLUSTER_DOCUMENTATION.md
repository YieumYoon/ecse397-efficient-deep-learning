# Markov GPU Cluster - Complete Documentation

This document summarizes the cluster verification and best practices based on official HPC CWRU documentation.

## 📋 Table of Contents
1. [Cluster Verification Results](#cluster-verification-results)
2. [Critical Best Practices](#critical-best-practices)
3. [Corrected Scripts Location](#corrected-scripts-location)
4. [How to Test Your Setup](#how-to-test-your-setup)
5. [Migration Guide](#migration-guide)

---

## Cluster Verification Results

### ✅ Verified Information (Tested on Sept 30, 2025)

| Component | Status | Details |
|-----------|--------|---------|
| **Partitions** | ✓ Verified | `markov_cpu*` (default), `markov_gpu` |
| **PyTorch Module** | ✓ Available | `PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1` |
| **Scratch Space** | ✓ Accessible | `/mnt/fs1` for Markov GPU nodes |
| **GPU Nodes** | ✓ Available | Multiple nodes (classt01-classt25) |
| **Module System** | ✓ Working | `module purge` and `module load` functional |

### Partition Details (from `sinfo`)

```bash
PARTITION      AVAIL  TIMELIMIT  NODES  STATE
markov_cpu*    up     13-08:00:0   29   various states
markov_gpu     up     13-08:00:0   23   various states
```

- **markov_cpu**: Default partition, CPU-only compute
- **markov_gpu**: GPU nodes with CUDA support

---

## Critical Best Practices

### 🚨 MUST DO: Use Scratch Space

**WRONG (violates HPC policy):**
```bash
cd /home/jxl2244/ecse397-efficient-deep-learning
python train.py  # Running in home directory - DON'T DO THIS!
```

**CORRECT (follows official guidelines):**
```bash
# Create unique work directory in scratch
WORK_DIR="/mnt/fs1/$USER/job_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Copy code to scratch
cp -r $HOME/project/code .

# Run computation in scratch
python train.py

# Copy results back
cp results/* $HOME/project/results/

# Cleanup
cd /mnt/fs1
rm -rf "$WORK_DIR"
```

### Why Scratch Space Matters

1. **Performance**: 10-100x faster I/O than home directory
2. **Reliability**: Prevents quota issues and job failures
3. **Cluster Health**: Reduces network load on shared filesystem
4. **Policy Compliance**: Required by official HPC guidelines

---

## Corrected Scripts Location

All corrected scripts are in: **`scripts_corrected/`**

### Available Scripts

| Script | Purpose | Time Limit | GPU |
|--------|---------|------------|-----|
| `train_cnn.slurm` | Train ResNet-18 on CIFAR-10 | 24h | 1 |
| `train_vit.slurm` | Train ViT-Tiny on CIFAR-10 | 24h | 1 |
| `prune_cnn.slurm` | Prune trained ResNet-18 | 12h | 1 |
| `prune_vit.slurm` | Prune trained ViT-Tiny | 12h | 1 |
| `test_cluster_setup.slurm` | Verify cluster setup | 10m | 1 |

### Key Improvements in Corrected Scripts

1. ✅ Use `/mnt/fs1` scratch space
2. ✅ Create unique work directories per job
3. ✅ Copy code to scratch before running
4. ✅ Copy results back to home before cleanup
5. ✅ Clean up scratch space after completion
6. ✅ Proper module loading with `module purge`
7. ✅ Error handling with `set -euo pipefail`
8. ✅ Informative logging and status messages

---

## How to Test Your Setup

### Step 1: Run Cluster Verification

```bash
# Submit test job
sbatch scripts_corrected/test_cluster_setup.slurm

# Check job status
squeue -u $USER

# View results (after job completes)
cat logs/test_cluster_*.out
```

The test verifies:
- ✓ Scratch space access and I/O
- ✓ Module system functionality
- ✓ Python and PyTorch installation
- ✓ CUDA/GPU availability
- ✓ GPU computation
- ✓ Data download (CIFAR-10)
- ✓ File copying operations
- ✓ Cleanup procedures

### Step 2: Run a Quick Training Test

```bash
# Test with 1 epoch only
EPOCHS=1 sbatch scripts_corrected/train_cnn.slurm

# Monitor progress
tail -f logs/train_cnn_*.out
```

### Step 3: Verify Results

```bash
# Check that checkpoint was saved
ls -lh pruning_lab/models_saved/

# Verify scratch was cleaned
# (should show no job directories for your user)
ls /mnt/fs1/$USER/
```

---

## Migration Guide

### From Old Scripts to Corrected Scripts

#### Option 1: Replace Scripts (Recommended)
```bash
# Backup old scripts
mv scripts scripts_old_backup

# Use corrected scripts
mv scripts_corrected scripts
```

#### Option 2: Use Corrected Scripts Directly
```bash
# Just submit from corrected directory
sbatch scripts_corrected/train_cnn.slurm
```

### Updating Custom Scripts

If you have custom scripts, update them using this template:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=HH:MM:SS
#SBATCH --output=/path/to/logs/job_%j.out
#SBATCH --error=/path/to/logs/job_%j.err

set -euo pipefail

# 1. Setup scratch space
WORK_DIR="/mnt/fs1/$USER/job_$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# 2. Copy required files
cp -r $HOME/myproject/code .
cp $HOME/myproject/data/checkpoint.pth .

# 3. Load modules
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# 4. Run your code
python train.py

# 5. Copy results back
cp -r outputs $HOME/myproject/results/

# 6. Cleanup
cd /mnt/fs1
rm -rf "$WORK_DIR"
```

---

## Quick Reference Commands

### Job Submission
```bash
# Submit job
sbatch script.slurm

# Submit with environment variables
EPOCHS=100 sbatch script.slurm

# Submit with specific partition
sbatch --partition=markov_gpu script.slurm
```

### Job Monitoring
```bash
# List your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Cancel job
scancel <job_id>

# View live output
tail -f logs/job_*.out
```

### Cluster Information
```bash
# List partitions
sinfo

# List available modules
module avail PyTorch

# Check disk quota
quota -s

# Check scratch space usage
du -sh /mnt/fs1/$USER/
```

---

## Additional Resources

1. **`.cursorrules`**: Complete guidelines for AI coding assistants
2. **`scripts_corrected/README.md`**: Detailed script documentation
3. **Official HPC Docs**: Contact CWRU HPC support for latest documentation

---

## Troubleshooting

### Job Fails with "Quota Exceeded"
- **Cause**: Running in home directory instead of scratch
- **Fix**: Use corrected scripts that use `/mnt/fs1`

### Job Fails with "Module not found"
- **Cause**: Missing `module purge` or incorrect module name
- **Fix**: Add `module purge` before `module load`

### Results Not Saved
- **Cause**: Forgot to copy results back before cleanup
- **Fix**: Always copy results to home before `rm -rf $WORK_DIR`

### Scratch Space Full
- **Cause**: Previous jobs didn't clean up
- **Fix**: Manually clean `/mnt/fs1/$USER/` and ensure scripts have cleanup

---

## Summary Checklist

When creating or reviewing SLURM scripts:

- [ ] Shebang: `#!/bin/bash`
- [ ] Error handling: `set -euo pipefail`
- [ ] Correct partition: `--partition=markov_gpu` for GPU jobs
- [ ] GPU request: `--gres=gpu:1` if needed
- [ ] Scratch space: `cd /mnt/fs1` and create unique work directory
- [ ] Copy code: `cp -r $HOME/project .`
- [ ] Load modules: `module purge` then `module load`
- [ ] Copy results: `cp results $HOME/` before cleanup
- [ ] Cleanup: `rm -rf $WORK_DIR` at end
- [ ] Logs: Output to home directory, not scratch

---

**Last Updated**: September 30, 2025  
**Verified Against**: Official CWRU HPC Documentation  
**Cluster**: markov.case.edu
