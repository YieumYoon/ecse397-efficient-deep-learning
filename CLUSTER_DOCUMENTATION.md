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
| **Scratch Space** | ✓ Accessible | `$TMPDIR` for jobs (`/tmp/job.JOBID.markov2`), `$PFSDIR` alternative |
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
# Use job-local scratch space (automatically provided by SLURM)
WORK_DIR="$TMPDIR"
cd "$WORK_DIR"

# Copy code to scratch
cp -r $HOME/project/code .

# Run computation in scratch
python train.py

# Copy results back
cp results/* $HOME/project/results/

# Cleanup: $TMPDIR is auto-cleaned by SLURM after the job ends
echo "Job complete - TMPDIR will be auto-cleaned"
```

### Why Scratch Space Matters

1. **Performance**: 10-100x faster I/O than home directory
2. **Reliability**: Prevents quota issues and job failures
3. **Cluster Health**: Reduces network load on shared filesystem
4. **Policy Compliance**: Required by official HPC guidelines

---

## Corrected Scripts Location

All corrected scripts are in: **`pruning_lab/utils/`**

### Available Scripts

| Script | Purpose | Time Limit | GPU |
|--------|---------|------------|-----|
| `train_cnn.slurm` | Train ResNet-18 on CIFAR-10 | 24h | 1 |
| `train_vit.slurm` | Train ViT-Tiny on CIFAR-10 | 24h | 1 |
| `prune_cnn.slurm` | Prune trained ResNet-18 | 12h | 1 |
| `prune_vit.slurm` | Prune trained ViT-Tiny | 12h | 1 |
| `test_cluster_setup.slurm` | Verify cluster setup | 10m | 1 |

### Key Improvements in Corrected Scripts

1. ✅ Use `$TMPDIR` job-local scratch space (unique per job)
2. ✅ No manual scratch directory management needed
3. ✅ Copy code to scratch before running
4. ✅ Copy results back to home before cleanup
5. ✅ No manual cleanup required ($TMPDIR auto-cleaned by SLURM)
6. ✅ Proper module loading with `module purge`
7. ✅ Error handling with `set -euo pipefail`
8. ✅ Informative logging and status messages

---

## How to Test Your Setup

### Step 1: Run Cluster Verification

```bash
# Submit test job
sbatch pruning_lab/utils/test_cluster_setup.slurm

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
EPOCHS=1 sbatch pruning_lab/utils/train_cnn.slurm

# Monitor progress
tail -f logs/train_cnn_*.out
```

### Step 3: Verify Results

```bash
# Check that checkpoint was saved
ls -lh pruning_lab/models_saved/

# Scratch space is job-local ($TMPDIR) and auto-cleaned by SLURM
# Verify results instead of checking $TMPDIR from login node
```

---

## Migration Guide

### From Old Scripts to Corrected Scripts

All scripts have already been consolidated into `pruning_lab/utils/`. If you had local
copies of older scripts, prefer the versions in `pruning_lab/utils/` and archive or delete
your local duplicates.

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

# 1. Use job-local scratch space
WORK_DIR="$TMPDIR"
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

# 6. Cleanup: $TMPDIR is auto-cleaned by SLURM
echo "TMPDIR will be auto-cleaned"
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

# Inspect job-local scratch space from within a job
# df -h "$TMPDIR"
```

---

## Additional Resources

1. **`.cursorrules`**: Complete guidelines for AI coding assistants
2. **`pruning_lab/utils/README.md`**: Detailed script documentation
3. **Official HPC Docs**: Contact CWRU HPC support for latest documentation

---

## Troubleshooting

### Job Fails with "Quota Exceeded"
- **Cause**: Running in home directory instead of scratch
- **Fix**: Use `$TMPDIR` as job-local scratch (provided by SLURM)

### Job Fails with "Module not found"
- **Cause**: Missing `module purge` or incorrect module name
- **Fix**: Add `module purge` before `module load`

### Results Not Saved
- **Cause**: Forgot to copy results back before cleanup
- **Fix**: Always copy results to home before `rm -rf $WORK_DIR`

### Scratch Space Full
- **Cause**: Node-local scratch ($TMPDIR) capacity reached
- **Fix**: Reduce dataset/temp footprint, shorten job, or try another time/node. Consider `$PFSDIR` if provided.

---

## Summary Checklist

When creating or reviewing SLURM scripts:

- [ ] Shebang: `#!/bin/bash`
- [ ] Error handling: `set -euo pipefail`
- [ ] Correct partition: `--partition=markov_gpu` for GPU jobs
- [ ] GPU request: `--gres=gpu:1` if needed
- [ ] Scratch space: `cd "$TMPDIR"` (auto-provided per job by SLURM)
- [ ] Copy code: `cp -r $HOME/project .`
- [ ] Load modules: `module purge` then `module load`
- [ ] Copy results: `cp results $HOME/` before cleanup
- [ ] Cleanup: `rm -rf $WORK_DIR` at end
- [ ] Logs: Output to home directory, not scratch

---

**Last Updated**: September 30, 2025  
**Verified Against**: Official CWRU HPC Documentation  
**Cluster**: markov.case.edu

---

## GPU Usage and Selection (Comprehensive)

### Available GPUs (Performance Ranking)

| GPU Type | Model | VRAM | Nodes | Performance | Best For |
|----------|-------|------|-------|-------------|----------|
| `gpu2h100` | NVIDIA H100 | ~80GB | 2 | ⭐⭐⭐⭐⭐ Best | Large models, fastest training |
| `gpu4090` | NVIDIA RTX 4090 | 24GB | 1 | ⭐⭐⭐⭐ Excellent | High-performance training |
| `gpul40s` | NVIDIA L40S | 48GB | 2 | ⭐⭐⭐⭐ Excellent | Large batches, good VRAM |
| `gpu4070` | NVIDIA RTX 4070 | 12GB | 1 | ⭐⭐⭐ Good | Standard training |
| `gpu2080` | RTX 2080 Ti | 11GB | 14+ | ⭐⭐⭐ Good | Most available, reliable |

### Check Availability

```bash
si                             # Show all GPU nodes and status
si | grep markov_gpu           # Filter GPU partition
si | grep -E "idle|mix"         # Focus on usable nodes
```

Interpreting columns: STATE = idle (best), mix (ok), alloc/down/drain (unusable). CPUS(A/I/O/T) shows allocated/idle CPU cores.

### Automatic GPU Selection (Recommended)

```bash
# Submit with auto-selected best GPU (H100 > 4090 > L40S > 4070 > 2080)
bash pruning_lab/utils/submit_best_gpu.sh pruning_lab/utils/train_cnn.slurm

# With environment variables
EPOCHS=100 bash pruning_lab/utils/submit_best_gpu.sh pruning_lab/utils/train_cnn.slurm
```

Under the hood, `pruning_lab/utils/select_best_gpu.sh` parses `si` output and returns the best available GPU feature. The wrapper adds `-C <feature>` to `sbatch`.

### Manual GPU Selection

```bash
# Request specific GPU type
sbatch -C gpu2h100 pruning_lab/utils/train_cnn.slurm   # H100 (fastest)
sbatch -C gpu4090  pruning_lab/utils/train_cnn.slurm   # RTX 4090
sbatch -C gpu2080  pruning_lab/utils/train_cnn.slurm   # RTX 2080 Ti (most available)
```

Add a constraint directly in a `.slurm` script:

```bash
#SBATCH -C gpu2h100   # or gpu4090, gpul40s, gpu4070, gpu2080
```

### Usage Examples

```bash
# 1) Auto-select best GPU
bash pruning_lab/utils/submit_best_gpu.sh pruning_lab/utils/train_cnn.slurm

# 2) Check then submit
GPU=$(bash pruning_lab/utils/select_best_gpu.sh)
sbatch -C "$GPU" pruning_lab/utils/train_cnn.slurm

# 3) Try H100, fallback to 2080 Ti
sbatch -C gpu2h100 pruning_lab/utils/train_cnn.slurm || sbatch -C gpu2080 pruning_lab/utils/train_cnn.slurm
```

### Tips

```bash
# Helpful aliases (put in ~/.bashrc)
alias gpu_status='si | grep markov_gpu | grep -E "idle|mix"'
alias gpu_h100='si | grep gpu2h100'
alias gpubest='bash ~/ecse397-efficient-deep-learning/pruning_lab/utils/select_best_gpu.sh'
```

---

## Quick Reference (One Page)

### Submit Jobs

```bash
# Auto-select best GPU (recommended)
bash pruning_lab/utils/submit_best_gpu.sh pruning_lab/utils/train_cnn.slurm

# Manual GPU choice
sbatch -C gpu2h100 pruning_lab/utils/train_cnn.slurm
sbatch -C gpu2080  pruning_lab/utils/train_cnn.slurm
```

### Monitor and Logs

```bash
squeue -u $USER
scontrol show job <job_id>

# Live logs
tail -f logs/train_cnn_*.out
```

### Common Workflows

```bash
# Quick test (1 epoch)
EPOCHS=1 bash pruning_lab/utils/submit_best_gpu.sh pruning_lab/utils/train_cnn.slurm

# Pruning
PRUNE_TYPE=unstructured AMOUNT=0.7 \
  bash pruning_lab/utils/submit_best_gpu.sh pruning_lab/utils/prune_cnn.slurm
```

### Scratch Space Reminder

```bash
# In SLURM jobs, use job-local scratch
cd "$TMPDIR"   # auto-created, auto-cleaned
```
