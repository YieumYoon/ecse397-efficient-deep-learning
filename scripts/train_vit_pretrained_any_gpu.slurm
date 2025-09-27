#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=24g
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=vit_tiny_pretrained
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
pip install timm --no-input

DATA_DIR="$TMP_BASE/data"
mkdir -p "$DATA_DIR"
OUT_DIR="runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

python -u -m pruning_lab.main train \
  --model vit_tiny_pretrained \
  --img-size 224 \
  --epochs 200 \
  --batch-size 128 \
  --optimizer adamw \
  --lr 5e-4 \
  --scheduler cosine \
  --t-max 200 \
  --amp \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT_DIR"


