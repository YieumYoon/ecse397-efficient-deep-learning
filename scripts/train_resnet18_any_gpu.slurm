#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=16g
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=resnet18
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

DATA_DIR="$TMP_BASE/data"
mkdir -p "$DATA_DIR"
OUT_DIR="runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

# Hyperparameters via environment overrides
EPOCHS=${EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-0.01}
OPTIMIZER=${OPTIMIZER:-sgd}
SCHEDULER=${SCHEDULER:-cosine}
MILESTONES=${MILESTONES:-"150 225"}
GAMMA=${GAMMA:-0.1}
T_MAX=${T_MAX:-$EPOCHS}
PRETRAINED=${PRETRAINED:-1}
AMP=${AMP:-1}
SEED=${SEED:-}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0005}
MOMENTUM=${MOMENTUM:-0.9}

ARGS=(
  -u -m pruning_lab.main train
  --model resnet18
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --optimizer "$OPTIMIZER"
  --lr "$LR"
  --scheduler "$SCHEDULER"
  --weight-decay "$WEIGHT_DECAY"
  --momentum "$MOMENTUM"
  --data-dir "$DATA_DIR"
  --output-dir "$OUT_DIR"
)

if [ "$SCHEDULER" = "multistep" ]; then
  # shellcheck disable=SC2086
  ARGS+=( --milestones $MILESTONES --gamma "$GAMMA" )
fi
if [ "$SCHEDULER" = "cosine" ]; then
  ARGS+=( --t-max "$T_MAX" )
fi
if [ "$PRETRAINED" = "1" ]; then
  ARGS+=( --pretrained )
fi
if [ "$AMP" = "1" ]; then
  ARGS+=( --amp )
fi
if [ -n "$SEED" ]; then
  ARGS+=( --seed "$SEED" )
fi

python "${ARGS[@]}"


