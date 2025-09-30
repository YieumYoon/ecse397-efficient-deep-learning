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
│   └── models_saved/               # Model checkpoints (git-ignored)
│
├── scripts/                        # Corrected HPC job scripts
│   ├── train_cnn.slurm
│   ├── train_vit.slurm
│   ├── prune_cnn.slurm
│   ├── prune_vit.slurm
│   ├── test_cluster_setup.slurm
│   ├── select_best_gpu.sh
│   └── submit_best_gpu.sh
│
├── logs/                           # Job logs (git-ignored)
│
├── requirements.txt                # Python deps (local dev)
├── Lab-1-2.md                      # Assignment handout
├── CLUSTER_DOCUMENTATION.md        # Comprehensive Markov cluster guide
└── README.md                       # This file
```

---

## Setup (Local)

```bash
pip install -r requirements.txt
python3 -m pruning_lab.main --help
```

---

## Markov GPU Cluster Quickstart

```bash
# SSH (VPN if off-campus)
ssh markov.case.edu
cd /home/jxl2244/ecse397-efficient-deep-learning

# Auto-select best GPU and submit training
bash scripts/submit_best_gpu.sh scripts/train_cnn.slurm

# Quick test (1 epoch)
EPOCHS=1 bash scripts/submit_best_gpu.sh scripts/train_cnn.slurm

# Verify cluster setup
sbatch scripts/test_cluster_setup.slurm
```

Notes
- Scripts run in $TMPDIR (job-local scratch, auto-cleaned by SLURM)
- To pin GPU type: `sbatch -C gpu2h100 scripts/train_cnn.slurm`
- Full cluster details in `CLUSTER_DOCUMENTATION.md`

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

- Module issues: load `PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1`, `pip install --user timm`
- CUDA OOM: reduce `--batch-size`
- SLURM failures: check `logs/`, `scontrol show job <job_id>`

---

## Docs

- Cluster guide: `CLUSTER_DOCUMENTATION.md`
- SLURM scripts and helpers: `scripts/`
