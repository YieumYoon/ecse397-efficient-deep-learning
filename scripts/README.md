# Corrected SLURM Scripts - Following Official HPC Documentation

This directory contains SLURM scripts that **correctly follow** the official CWRU HPC best practices for the Markov GPU cluster.

## Key Differences from Original Scripts

### ❌ Original Scripts (`scripts/`)
The original scripts had several critical violations of HPC guidelines:

1. **Ran in home directory** instead of scratch space
   ```bash
   cd /home/jxl2244/ecse397-efficient-deep-learning  # WRONG!
   ```

2. **No data copying** to/from scratch
3. **No cleanup** of scratch space
4. **Could cause I/O bottlenecks** and quota issues

### ✅ Corrected Scripts (`scripts_corrected/`)
These scripts follow all official guidelines:

1. **Use scratch space** (`$TMPDIR` for Markov jobs; `$PFSDIR` optional alternative)
   ```bash
   WORK_DIR="$TMPDIR"
   cd "$WORK_DIR"
   ```

2. **Copy code to scratch** at job start
   ```bash
   cp -r $HOME/ecse397-efficient-deep-learning/pruning_lab .
   ```

3. **Copy results back** before job ends
   ```bash
   cp -v models_saved/* $HOME/ecse397-efficient-deep-learning/pruning_lab/models_saved/
   ```

4. **Clean up scratch space** after completion
   ```bash
   rm -rf "$WORK_DIR"
   ```

5. **Proper module loading** with purge
   ```bash
   module purge
   module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
   ```

## Verified Cluster Information

### Partitions (tested with `sinfo`)
- **markov_cpu*** (default) - CPU-only compute nodes
- **markov_gpu** - GPU nodes with CUDA support

### Available Modules (tested with `module avail`)
- **PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1** ✓ Available and working

### Scratch Space
- **Markov GPU nodes**: `$TMPDIR` (auto-provided per job by SLURM)
- **Other nodes**: `$TMPDIR` or `$PFSDIR` (only available in job context)

## How to Use These Scripts

### 1. Submit a Training Job
```bash
sbatch scripts_corrected/train_cnn.slurm
```

### 2. Submit with Custom Parameters
```bash
EPOCHS=100 LR=0.05 sbatch scripts_corrected/train_cnn.slurm
```

### 3. Submit a Pruning Job
```bash
PRUNE_TYPE=structured AMOUNT=0.5 sbatch scripts_corrected/prune_cnn.slurm
```

## Benefits of Using Scratch Space

1. **Faster I/O**: Scratch space is optimized for high-speed read/write
2. **No quota issues**: Home directory quotas won't be exceeded during computation
3. **Better performance**: Reduces network I/O load
4. **Cluster citizenship**: Follows best practices, doesn't slow down login nodes
5. **Job reliability**: Jobs less likely to fail due to I/O bottlenecks

## What Gets Copied

### To Scratch (at job start):
- Project code (`pruning_lab/`, `scripts/`)
- Pre-trained checkpoints (for pruning jobs only)

### NOT Copied (downloaded automatically by PyTorch):
- CIFAR-10 dataset (downloads to scratch as needed)
- Pre-trained model weights (downloads as needed)

### From Scratch (at job end):
- Trained model checkpoints
- Any output files in `models_saved/`

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
scontrol show job <job_id>

# Cancel job
scancel <job_id>

# View logs (while job is running or after)
tail -f logs/train_cnn_<job_id>.out
tail -f logs/train_cnn_<job_id>.err
```

## Testing

Before running long jobs, test with shorter parameters:
```bash
EPOCHS=1 sbatch scripts_corrected/train_cnn.slurm
```

## Migration from Old Scripts

If you've been using the old scripts successfully, it may be because:
1. Your jobs were small enough to not hit I/O limits
2. You got lucky with node load
3. The cluster admins haven't enforced the policy yet

**However**, following best practices ensures:
- Your jobs will always work reliably
- You're a good cluster citizen
- Your jobs run faster with better I/O performance

## Additional Resources

- Official documentation: See `.cursorrules` in repository root
- Check available modules: `module avail PyTorch`
- Check partition info: `sinfo -o "%P %.5a %.10l %.6D %.6t %N"`
- Check your quota: `quota -s`
