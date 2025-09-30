# Python Environment Setup for Markov HPC

## Quick Setup (2 steps)

### 1. Run Setup Script (once)

```bash
cd /home/jxl2244/ecse397-efficient-deep-learning
bash scripts/setup_env.sh
```

This loads PyTorch from HPC modules and installs timm (~6MB).

### 2. Submit Jobs

```bash
# Submit all training and pruning jobs with dependencies
bash scripts/submit_all.sh
```

Done! SLURM scripts load the environment automatically.

---

## What's Installed

- Python 3.11.5 (HPC module)
- PyTorch 2.1.2 + CUDA 12.1 (HPC module)
- torchvision 0.16.2 (HPC module)
- timm 1.0.20 (installed to ~/.local)

**Total downloads: ~6MB** (no disk quota issues!)

---

## For Interactive Use

```bash
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
python3
```

---

## Troubleshooting

**Module not found?**
```bash
module avail 2>&1 | grep -i pytorch
```

**Import errors?**
```bash
module list  # Check what's loaded
```

**Jobs failing?**
```bash
cat <jobname>_*.err  # Check error log
```
