#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=24g
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --job-name=prune
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
# Arguments (pass via sbatch --export or environment vars):
#   MODEL:                one of {resnet18, vit_tiny_pretrained, vit_tiny_scratch}
#   PRUNE_TYPE:           one of {unstructured, structured}
#   AMOUNTS:              space-separated list, used with array jobs (e.g., "0.5 0.9")
#   CHECKPOINT:           path to trained checkpoint to prune
#   FINETUNE_EPOCHS:      epochs to fine-tune after pruning (default 20)
#   LR:                   learning rate (default 0.01 for resnet/structured; 5e-4 for vit/unstructured)
#   OPTIMIZER:            one of {sgd, adamw} (defaults based on model)
#   SCHEDULER:            one of {none, multistep, cosine} (default cosine)
#   T_MAX:                cosine T_max (default equals FINETUNE_EPOCHS)
#   BATCH_SIZE:           fine-tune batch size (default 128)
#   DATA_DIR:             dataset cache directory (default $TMP_BASE/data)
#   IMG_SIZE:             override image size (default from model)
#   EXTRA_FLAGS:          any extra flags to forward to CLI
#

MODEL=${MODEL:-resnet18}
PRUNE_TYPE=${PRUNE_TYPE:-unstructured}
AMOUNTS_STR=${AMOUNTS:-"0.5 0.9"}
CHECKPOINT=${CHECKPOINT:-""}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-20}
LR=${LR:-""}
OPTIMIZER=${OPTIMIZER:-""}
SCHEDULER=${SCHEDULER:-cosine}
T_MAX=${T_MAX:-$FINETUNE_EPOCHS}
BATCH_SIZE=${BATCH_SIZE:-128}
DATA_DIR=${DATA_DIR:-"$TMP_BASE/data"}
IMG_SIZE=${IMG_SIZE:-""}
EXTRA_FLAGS=${EXTRA_FLAGS:-""}

mkdir -p "$DATA_DIR"
OUT_DIR="runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

# Derive sensible defaults based on model if not provided
if [ -z "$LR" ]; then
  case "$MODEL" in
    vit_tiny_pretrained|vit_tiny_scratch)
      LR=5e-4
      ;;
    *)
      LR=0.01
      ;;
  esac
fi

if [ -z "$OPTIMIZER" ]; then
  case "$MODEL" in
    vit_tiny_pretrained|vit_tiny_scratch)
      OPTIMIZER=adamw
      ;;
    *)
      OPTIMIZER=sgd
      ;;
  esac
fi

# Determine default img size for ViT if unspecified
MODEL_IMG_FLAGS=""
case "$MODEL" in
  vit_tiny_pretrained|vit_tiny_scratch)
    if [ -z "$IMG_SIZE" ]; then
      MODEL_IMG_FLAGS="--img-size 224"
    else
      MODEL_IMG_FLAGS="--img-size $IMG_SIZE"
    fi
    ;;
  *)
    if [ -n "$IMG_SIZE" ]; then
      MODEL_IMG_FLAGS="--img-size $IMG_SIZE"
    fi
    ;;
esac

if [ -z "$CHECKPOINT" ]; then
  echo "ERROR: CHECKPOINT must be provided"
  exit 1
fi

# Support both regular and array jobs. If array, pick amount by index.
IFS=' ' read -r -a AMOUNTS_ARR <<< "$AMOUNTS_STR"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  IDX=${SLURM_ARRAY_TASK_ID}
  if [ "$IDX" -lt 0 ] || [ "$IDX" -ge "${#AMOUNTS_ARR[@]}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=$IDX for AMOUNTS=[$AMOUNTS_STR]"
    exit 1
  fi
  AMOUNT="${AMOUNTS_ARR[$IDX]}"
else
  # If not an array job, use the first amount
  AMOUNT="${AMOUNTS_ARR[0]}"
fi

echo "Pruning MODEL=$MODEL PRUNE_TYPE=$PRUNE_TYPE AMOUNT=$AMOUNT from CHECKPOINT=$CHECKPOINT"

# Choose extra prune flags based on type
PRUNE_FLAGS=""
if [ "$PRUNE_TYPE" = "unstructured" ]; then
  # Optionally include bias/norm for experiments; default conservative for CNNs
  if [[ "$MODEL" == vit_* ]]; then
    PRUNE_FLAGS="--include-norm"
  else
    PRUNE_FLAGS="--include-bias"
  fi
fi

python -u -m pruning_lab.main prune \
  --model "$MODEL" \
  $MODEL_IMG_FLAGS \
  --checkpoint "$CHECKPOINT" \
  --prune-type "$PRUNE_TYPE" \
  --amount "$AMOUNT" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --scheduler "$SCHEDULER" \
  --t-max "$T_MAX" \
  --data-dir "$DATA_DIR" \
  --output-checkpoint "$OUT_DIR/${MODEL}_${PRUNE_TYPE}_amount${AMOUNT}.pth" \
  $PRUNE_FLAGS \
  $EXTRA_FLAGS

echo "Done. Output checkpoint in $OUT_DIR"


