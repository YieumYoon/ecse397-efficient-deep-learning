# Quick Reference Card - Markov GPU Cluster

## 🚀 Most Common Commands

### Check GPU Availability
```bash
si                              # Show all GPU nodes and status
si | grep markov_gpu            # Show only GPU nodes
si | grep idle                  # Show only idle nodes
```

### Submit Jobs

#### Auto-Select Best GPU (Recommended!)
```bash
# Automatic GPU selection and submission
bash scripts/submit_best_gpu.sh scripts/train_cnn.slurm

# With environment variables
EPOCHS=100 bash scripts/submit_best_gpu.sh scripts/train_cnn.slurm
```

#### Manual GPU Selection
```bash
# Request specific GPU
sbatch -C gpu2h100 scripts/train_cnn.slurm    # H100 (fastest)
sbatch -C gpu4090 scripts/train_cnn.slurm     # RTX 4090
sbatch -C gpu2080 scripts/train_cnn.slurm     # RTX 2080 Ti (most available)

# Default (no GPU preference)
sbatch scripts/train_cnn.slurm
```

### Monitor Jobs
```bash
squeue -u $USER                 # Check your jobs
scontrol show job <job_id>      # Detailed job info
scancel <job_id>                # Cancel a job
scancel -u $USER                # Cancel all your jobs
```

### View Logs
```bash
# Live monitoring
tail -f logs/train_cnn_*.out    # Watch training output
tail -f logs/train_cnn_*.err    # Watch errors

# After completion
cat logs/train_cnn_12345.out    # View full output
cat logs/train_cnn_12345.err    # View errors
```

---

## 🎮 Available GPUs (Ranked)

| GPU | Type | VRAM | Speed | Availability | Command |
|-----|------|------|-------|--------------|---------|
| 🥇 H100 | `gpu2h100` | 80GB | ⚡⚡⚡⚡⚡ | 2 nodes | `-C gpu2h100` |
| 🥈 RTX 4090 | `gpu4090` | 24GB | ⚡⚡⚡⚡ | 1 node | `-C gpu4090` |
| 🥉 L40S | `gpul40s` | 48GB | ⚡⚡⚡⚡ | 2 nodes | `-C gpul40s` |
| RTX 4070 | `gpu4070` | 12GB | ⚡⚡⚡ | 1 node | `-C gpu4070` |
| RTX 2080 Ti | `gpu2080` | 11GB | ⚡⚡⚡ | 14+ nodes | `-C gpu2080` |

---

## 📋 Common Workflows

### 1. Quick Test (1 epoch)
```bash
EPOCHS=1 bash scripts/submit_best_gpu.sh scripts/train_cnn.slurm
```

### 2. Full Training
```bash
bash scripts/submit_best_gpu.sh scripts/train_cnn.slurm
```

### 3. Pruning
```bash
# Unstructured pruning (70%)
PRUNE_TYPE=unstructured AMOUNT=0.7 \
  bash scripts/submit_best_gpu.sh scripts/prune_cnn.slurm

# Structured pruning (50%)
PRUNE_TYPE=structured AMOUNT=0.5 \
  bash scripts/submit_best_gpu.sh scripts/prune_cnn.slurm
```

### 4. Custom Training Parameters
```bash
EPOCHS=100 LR=0.05 BATCH_SIZE=256 \
  bash scripts/submit_best_gpu.sh scripts/train_cnn.slurm
```

---

## 📊 Job Status Meanings

| Status | Meaning | Action |
|--------|---------|--------|
| `PD` | Pending (waiting) | Wait or check if resources available |
| `R` | Running | Monitor with `tail -f logs/*` |
| `CG` | Completing | Almost done |
| `CD` | Completed | Check logs for results |
| `F` | Failed | Check error logs |
| `CA` | Cancelled | Job was cancelled |

---

## 🛠️ Troubleshooting

### Job Stuck in Queue?
```bash
# Check why job is pending
squeue -u $USER
scontrol show job <job_id> | grep Reason

# Try different GPU if waiting too long
scancel <job_id>
sbatch -C gpu2080 scripts/train_cnn.slurm  # Most available
```

### Check Results
```bash
# List saved models
ls -lh pruning_lab/models_saved/

# Check log files
ls -lt logs/ | head -10
```

### Disk Usage
```bash
quota -s                        # Check your quota
du -sh pruning_lab/            # Check directory size
```

---

## 🎯 Available Scripts

| Script | Purpose | GPU | Time |
|--------|---------|-----|------|
| `train_cnn.slurm` | Train ResNet-18 | 1 | 24h |
| `train_vit.slurm` | Train ViT-Tiny | 1 | 24h |
| `prune_cnn.slurm` | Prune ResNet-18 | 1 | 12h |
| `prune_vit.slurm` | Prune ViT-Tiny | 1 | 12h |
| `test_cluster_setup.slurm` | Verify setup | 1 | 10m |

All scripts in: `scripts/`

---

## 💡 Pro Tips

### Alias Setup (Add to ~/.bashrc)
```bash
alias gpu='si | grep markov_gpu'
alias myjobs='squeue -u $USER'
alias logs='ls -lt logs/ | head -10'
alias gpubest='bash ~/ecse397-efficient-deep-learning/scripts/select_best_gpu.sh'
```

### Quick Submit Function (Add to ~/.bashrc)
```bash
qsub() {
    bash ~/ecse397-efficient-deep-learning/scripts/submit_best_gpu.sh "$@"
}
```

Then use: `qsub scripts/train_cnn.slurm`

---

## 📞 Quick Help

| Need | Command | File |
|------|---------|------|
| GPU info | `si` | - |
| Best GPU | `bash scripts/select_best_gpu.sh` | - |
| Submit auto | `bash scripts/submit_best_gpu.sh <script>` | - |
| Job status | `squeue -u $USER` | - |
| Full guide | - | `GPU_SELECTION_GUIDE.md` |
| Setup info | - | `SETUP_VERIFIED.md` |
| Migration | - | `MIGRATION_GUIDE.md` |
| Documentation | - | `CLUSTER_DOCUMENTATION.md` |

---

## 🔥 One-Liners

```bash
# Check GPU availability and submit in one go
si && bash scripts/submit_best_gpu.sh scripts/train_cnn.slurm

# Submit to best GPU if H100 available, otherwise wait
si | grep -q "gpu2h100.*idle" && sbatch -C gpu2h100 scripts/train_cnn.slurm

# Check job progress
watch -n 10 'squeue -u $USER && tail -20 logs/train_cnn_*.out 2>/dev/null'

# Find and view latest log
cat $(ls -t logs/*.out | head -1)
```

---

**TL;DR: Just run this:**
```bash
bash scripts/submit_best_gpu.sh scripts/train_cnn.slurm
```

It automatically picks the best available GPU and submits your job! 🚀

---

*Print this card and keep it handy!*  
*Last updated: September 30, 2025*
