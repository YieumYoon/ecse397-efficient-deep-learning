# ECSE 397/600: Efficient Deep Learning - Lab 1-2

Custom pruning of ResNet-18 and ViT-Tiny on CIFAR-10 with SLURM-ready scripts for the CWRU Markov GPU cluster.

---

## Repository Structure

```
ecse397-efficient-deep-learning/
│
├── pruning_lab/                    # Main submission folder
│   ├── __init__.py
│   ├── main.py                     # CLI entry point
│   ├── report.json                 # Final metrics report
|   ├── Lab-1-2.md                  # Assignment handout
|   ├── README.md                   # This file
│   │
│   ├── data/
│   │   └── dataloader.py           # CIFAR-10 loaders with augmentations (uses $TMPDIR in jobs)
│   │
│   ├── models/
│   │   ├── resnet18.py
│   │   └── vit_tiny.py
│   │
│   ├── train/
│   │   ├── train_loop.py
│   │   └── prune.py
│   │
│   ├── inference/
│   │   └── test.py
│   │
|   ├── utils/                     # HPC job scripts
│   |   ├── train_cnn.slurm
│   |   ├── train_vit.slurm
│   |   ├── prune_cnn.slurm
│   |   ├── prune_vit.slurm
│   |   └── prune_vit_scratch.slurm
│   │
│   │
│   └── models_saved/               # Model checkpoints
│
├── logs/                           # Job logs
│
├── requirements.txt                # Python deps (local dev)
└── CLUSTER_DOCUMENTATION.md        # Comprehensive Markov cluster guide
```

---

## Setup (Local)

```bash
pip install -r requirements.txt
python3 -m pruning_lab.main --help
```

---

## Setup (Markov, with .venv for modern GPUs)

Use a project-local virtual environment (`.venv`) with PyTorch 2.5.1 + CUDA 12.1. The SLURM scripts will copy and activate it automatically.

```bash
# Load modern Python and CUDA modules on login node
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

cd /home/$USER/ecse397-efficient-deep-learning

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install packages (PyTorch 2.5.1 with CUDA 12.1)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm tqdm

# Quick check
python -c 'import torch; print("PyTorch:", torch.__version__)'
```

Notes:
- `.venv` is git-ignored and ~3–5 GB; do not commit it.
- No extra flags needed; SLURM scripts handle activation.

## Markov GPU Cluster Quickstart

```bash
# SSH (VPN if off-campus)
ssh markov.case.edu
cd /home/jxl2244/ecse397-efficient-deep-learning

# Check GPU availability
si | grep markov_gpu

# Request specific GPU type (recommended)
sbatch -C gpu2h100 pruning_lab/utils/train_cnn.slurm  # H100 (fastest)
sbatch -C gpu4090 pruning_lab/utils/train_vit.slurm   # RTX 4090 (2nd fastest)

# Quick test (1 epoch)
EPOCHS=1 sbatch -C gpu2h100 pruning_lab/utils/train_cnn.slurm

# Verify cluster setup (if available)
# sbatch pruning_lab/utils/test_cluster_setup.slurm
```

### ⚠️ GPU Selection Best Practices

**CRITICAL:** Always specify GPU type to avoid slow nodes!

```bash
# ✅ GOOD: Request specific fast GPU
sbatch -C gpu2h100 pruning_lab/utils/train_cnn.slurm

# ❌ BAD: Random GPU assignment (might get slow gpu2080)
sbatch pruning_lab/utils/train_cnn.slurm
```

**GPU Performance Tiers:**
- `gpu2h100` (H100) - **FASTEST** - classt[23-24] (2 nodes)
- `gpu4090` (RTX 4090) - Excellent - classt25 (1 node)
- `gpul40s` (L40S) - Good - classt[21-22] (2 nodes)
- `gpu4070` (RTX 4070) - Decent - classt06 (1 node)
- `gpu2080` (RTX 2080 Ti) - **SLOWEST** - classt[01-19] (14+ nodes, most available)

**Performance Impact:**
- Training on H100: ~2 hours
- Training on gpu2080: ~5 hours (2.5x slower!)

**Check which GPU your job got:**
```bash
squeue -u $USER                        # See running jobs
sacct -j <job_id> --format=NodeList    # Check node assignment
# If classt13-19 (gpu2080), cancel and resubmit with -C gpu2h100
```

### Notes
- Scripts run in $TMPDIR (job-local scratch, auto-cleaned by SLURM)
- SLURM scripts use `.venv` (PyTorch 2.5.1+cu121) and copy/activate it on the node
- Full cluster details in `CLUSTER_DOCUMENTATION.md` and `.cursorrules`

---

## Usage (Local CLI)

### Train
```bash
# ResNet-18
python3 -m pruning_lab.main train \
  --model resnet18 \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --output-dir pruning_lab/models_saved \
  --checkpoint-name resnet18.pth

# ViT-Tiny (pretrained)
python3 -m pruning_lab.main train \
  --model vit_tiny_pretrained \
  --epochs 100 \
  --batch-size 64 \
  --lr 0.001 \
  --optimizer adamw \
  --scheduler cosine \
  --output-dir pruning_lab/models_saved
```

### Test
```bash
python3 -m pruning_lab.main test \
  --model resnet18 \
  --checkpoint pruning_lab/models_saved/resnet18.pth \
  --batch-size 256
```

### Prune
```bash
# Unstructured + fine-tune
python3 -m pruning_lab.main prune \
  --model resnet18 \
  --checkpoint pruning_lab/models_saved/cnn_before_pruning.pth \
  --prune-type unstructured \
  --amount 0.7 \
  --finetune-epochs 50 \
  --lr 0.01 \
  --output-checkpoint pruning_lab/models_saved/cnn_after_unstructured_pruning.pth

# Structured + fine-tune
python3 -m pruning_lab.main prune \
  --model resnet18 \
  --checkpoint pruning_lab/models_saved/cnn_before_pruning.pth \
  --prune-type structured \
  --amount 0.5 \
  --finetune-epochs 50 \
  --lr 0.01 \
  --output-checkpoint pruning_lab/models_saved/cnn_after_structured_pruning.pth
```

Models
- `resnet18`
- `vit_tiny_pretrained`
- `vit_tiny_scratch`

Key CLI Args
- Training: `--epochs`, `--batch-size`, `--lr`, `--optimizer {sgd|adamw}`, `--scheduler {none|multistep|cosine}`, `--amp`
- Pruning: `--prune-type {unstructured|structured}`, `--amount`, `--finetune-epochs`

---

## Job Management (HPC)

```bash
squeue -u $USER               # status
scontrol show job <job_id>    # details
scancel <job_id>              # cancel
```

---

## Results

See `pruning_lab/report.json` for metrics; checkpoints saved under `pruning_lab/models_saved/`.

---

## Troubleshooting

- **Slow training?** Check GPU type with `sacct -j <job_id> --format=NodeList`. If gpu2080 (classt01-19), cancel and resubmit with `-C gpu2h100`
- Environment issues:
  - Ensure these are loaded in jobs: `module load Python/3.11.3-GCCcore-12.3.0` and `module load CUDA/12.1.1`
  - Recreate venv if needed:
    ```bash
    rm -rf .venv
    module purge && module load Python/3.11.3-GCCcore-12.3.0 && module load CUDA/12.1.1
    python3 -m venv .venv && source .venv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install timm tqdm
    ```
- Disk quota exceeded: clean large caches then retry
  ```bash
  rm -rf ~/.cache/pip ~/.cache/torch ~/.cache/huggingface
  quota -s
  ```
- CUDA OOM: reduce `--batch-size`
- SLURM failures: check `logs/`, `scontrol show job <job_id>`
- Job stuck in queue: Try different GPU type or check `si` for availability

---

## Docs

- Cluster guide: `CLUSTER_DOCUMENTATION.md`
- SLURM scripts and helpers: `pruning_lab/utils/`
