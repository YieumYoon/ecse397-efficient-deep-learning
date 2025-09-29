#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=24g
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-16:00:00
#SBATCH --job-name=vit_pretrained_criteria
#SBATCH --output=%x-%j.out

set -euo pipefail

cd /home/jxl2244/ecse397-efficient-deep-learning

module purge || true
TMP_BASE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

# Install Miniforge (Conda) locally to get modern Python
if [ ! -d "$TMP_BASE/miniforge3" ]; then
  echo "Installing Miniforge under $TMP_BASE/miniforge3"
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O "$TMP_BASE/miniforge.sh"
  bash "$TMP_BASE/miniforge.sh" -b -p "$TMP_BASE/miniforge3"
fi

set +u
source "$TMP_BASE/miniforge3/etc/profile.d/conda.sh"
set -u
conda create -y -p "$TMP_BASE/py310" python=3.10
conda activate "$TMP_BASE/py310"
python -c 'import sys; print("Using Python:", sys.version)'
python -m pip install --upgrade pip

# GPU torch if possible, else CPU
if pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision --no-input; then
  echo "Installed CUDA-enabled torch"
else
  echo "Falling back to CPU torch"
  pip install torch torchvision --no-input
fi

# ViT dependency
pip install timm --no-input

#
# Environment variables:
#   START_CHECKPOINT      absolute path to ViT-Tiny pretrained fine-tuned checkpoint (required)
#   UNSTRUCTURED_AMOUNT   target global sparsity for unstructured (default 0.7)
#   STRUCTURED_AMOUNT     target channel sparsity for structured (default 0.25)
#   EPOCHS                fine-tune epochs for each prune (default 20)
#   BATCH_SIZE            fine-tune batch size (default 128)
#   LR                    fine-tune learning rate (default 5e-4)
#   OPTIMIZER             {adamw, sgd} (default adamw)
#   SCHEDULER             {none,multistep,cosine} (default cosine)
#   T_MAX                 cosine T_max (default = EPOCHS)
#   DATA_DIR              dataset cache directory (default $TMP_BASE/data)
#   EXTRA_FLAGS           extra flags forwarded to the CLI (e.g., "--amp")
#

MODEL=vit_tiny_pretrained
IMG_FLAGS=(--img-size 224)
START_CHECKPOINT=${START_CHECKPOINT:-""}
UNSTRUCTURED_AMOUNT=${UNSTRUCTURED_AMOUNT:-0.7}
STRUCTURED_AMOUNT=${STRUCTURED_AMOUNT:-0.25}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-5e-4}
OPTIMIZER=${OPTIMIZER:-adamw}
SCHEDULER=${SCHEDULER:-cosine}
T_MAX=${T_MAX:-$EPOCHS}
DATA_DIR=${DATA_DIR:-"$TMP_BASE/data"}
EXTRA_FLAGS=${EXTRA_FLAGS:-"--amp"}

mkdir -p "$DATA_DIR"
OUT_DIR="runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

if [ -z "$START_CHECKPOINT" ]; then
  echo "ERROR: START_CHECKPOINT must be provided"
  exit 1
fi

echo "Evaluating baseline: $START_CHECKPOINT"
python -u -m pruning_lab.main test \
  --model "$MODEL" \
  "${IMG_FLAGS[@]}" \
  --checkpoint "$START_CHECKPOINT" \
  --batch-size 256 \
  --data-dir "$DATA_DIR" \
  | tee "$OUT_DIR/baseline_test.json"

# ========== Unstructured pruning (≥ 70%) ==========
UNS_CKPT="$OUT_DIR/${MODEL}_unstructured_amount${UNSTRUCTURED_AMOUNT}.pth"
echo "Pruning unstructured to amount=${UNSTRUCTURED_AMOUNT} with finetune ${EPOCHS} epochs"
python -u -m pruning_lab.main prune \
  --model "$MODEL" \
  "${IMG_FLAGS[@]}" \
  --checkpoint "$START_CHECKPOINT" \
  --prune-type unstructured \
  --amount "$UNSTRUCTURED_AMOUNT" \
  --finetune-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --scheduler "$SCHEDULER" \
  --t-max "$T_MAX" \
  --data-dir "$DATA_DIR" \
  --output-checkpoint "$UNS_CKPT" \
  --include-norm \
  $EXTRA_FLAGS \
  | tee "$OUT_DIR/unstructured_prune.json"

python -u -m pruning_lab.main test \
  --model "$MODEL" \
  "${IMG_FLAGS[@]}" \
  --checkpoint "$UNS_CKPT" \
  --batch-size 256 \
  --data-dir "$DATA_DIR" \
  | tee "$OUT_DIR/unstructured_test.json"

# ========== Structured pruning (≥ 25%) ==========
STR_CKPT="$OUT_DIR/${MODEL}_structured_amount${STRUCTURED_AMOUNT}.pth"
echo "Pruning structured (channels) to amount=${STRUCTURED_AMOUNT} with finetune ${EPOCHS} epochs"
python -u -m pruning_lab.main prune \
  --model "$MODEL" \
  "${IMG_FLAGS[@]}" \
  --checkpoint "$START_CHECKPOINT" \
  --prune-type structured \
  --amount "$STRUCTURED_AMOUNT" \
  --finetune-epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --scheduler "$SCHEDULER" \
  --t-max "$T_MAX" \
  --data-dir "$DATA_DIR" \
  --output-checkpoint "$STR_CKPT" \
  $EXTRA_FLAGS \
  | tee "$OUT_DIR/structured_prune.json"

python -u -m pruning_lab.main test \
  --model "$MODEL" \
  "${IMG_FLAGS[@]}" \
  --checkpoint "$STR_CKPT" \
  --batch-size 256 \
  --data-dir "$DATA_DIR" \
  | tee "$OUT_DIR/structured_test.json"

# ========== Summarize to report ==========
python - << 'PY'
import json, os
out_dir = os.path.join('runs', f"{os.environ.get('SLURM_JOB_NAME', 'vit_pretrained_criteria')}-{os.environ.get('SLURM_JOB_ID', 'local')}")

def _read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

report = {
    'model': 'vit_tiny_pretrained',
    'criteria': {
        'accuracy_min': 0.88,
        'unstructured_sparsity_min': 0.70,
        'structured_sparsity_min': 0.25,
    },
    'baseline': _read(os.path.join(out_dir, 'baseline_test.json')),
    'unstructured': {
        'prune': _read(os.path.join(out_dir, 'unstructured_prune.json')),
        'test': _read(os.path.join(out_dir, 'unstructured_test.json')),
    },
    'structured': {
        'prune': _read(os.path.join(out_dir, 'structured_prune.json')),
        'test': _read(os.path.join(out_dir, 'structured_test.json')),
    },
}

with open(os.path.join(out_dir, 'vit_tiny_pretrained_criteria_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

print(json.dumps({'report_path': os.path.join(out_dir, 'vit_tiny_pretrained_criteria_report.json')}, indent=2))
PY



