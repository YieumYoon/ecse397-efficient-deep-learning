#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=24g
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --job-name=iter_prune
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
# Required/optional environment variables:
#   MODEL                {resnet18, vit_tiny_pretrained, vit_tiny_scratch}
#   PRUNE_TYPE           {unstructured, structured}
#   AMOUNTS              space-separated cumulative sparsity targets (e.g., "0.5 0.7 0.9")
#   START_CHECKPOINT     path to checkpoint to start from (before pruning)
#   PER_STEP_EPOCHS      fine-tune epochs per step (default 20)
#   LR                   fine-tuning learning rate (default model-based)
#   OPTIMIZER            {sgd, adamw} (default model-based)
#   SCHEDULER            {none, multistep, cosine} (default cosine)
#   T_MAX                cosine T_max (default = PER_STEP_EPOCHS)
#   BATCH_SIZE           fine-tuning batch size (default 128)
#   DATA_DIR             dataset cache directory (default $TMP_BASE/data)
#   IMG_SIZE             override image size (default from model)
#   EXTRA_FLAGS          extra flags forwarded to the CLI
#

MODEL=${MODEL:-resnet18}
PRUNE_TYPE=${PRUNE_TYPE:-unstructured}
AMOUNTS_STR=${AMOUNTS:-"0.5 0.7 0.9"}
START_CHECKPOINT=${START_CHECKPOINT:-""}
PER_STEP_EPOCHS=${PER_STEP_EPOCHS:-20}
LR=${LR:-""}
OPTIMIZER=${OPTIMIZER:-""}
SCHEDULER=${SCHEDULER:-cosine}
T_MAX=${T_MAX:-$PER_STEP_EPOCHS}
BATCH_SIZE=${BATCH_SIZE:-128}
DATA_DIR=${DATA_DIR:-"$TMP_BASE/data"}
IMG_SIZE=${IMG_SIZE:-""}
EXTRA_FLAGS=${EXTRA_FLAGS:-""}

mkdir -p "$DATA_DIR"
OUT_DIR="runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

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

if [ -z "$START_CHECKPOINT" ]; then
  echo "ERROR: START_CHECKPOINT must be provided"
  exit 1
fi

echo "Evaluating baseline: $START_CHECKPOINT"
python -u -m pruning_lab.main test \
  --model "$MODEL" \
  $MODEL_IMG_FLAGS \
  --checkpoint "$START_CHECKPOINT" \
  --batch-size 256 \
  --data-dir "$DATA_DIR" \
  | tee "$OUT_DIR/step_00_baseline_test.json"

CURRENT_CKPT="$START_CHECKPOINT"

IFS=' ' read -r -a AMOUNTS_ARR <<< "$AMOUNTS_STR"
STEP_IDX=1
for AMOUNT in "${AMOUNTS_ARR[@]}"; do
  echo "\n=== Iteration $STEP_IDX targeting global sparsity $AMOUNT ==="

  PRUNE_FLAGS=""
  if [ "$PRUNE_TYPE" = "unstructured" ]; then
    if [[ "$MODEL" == vit_* ]]; then
      PRUNE_FLAGS="--include-norm"
    else
      PRUNE_FLAGS="--include-bias"
    fi
  fi

  OUTPUT_CKPT="$OUT_DIR/${MODEL}_${PRUNE_TYPE}_amount${AMOUNT}.pth"

  python -u -m pruning_lab.main prune \
    --model "$MODEL" \
    $MODEL_IMG_FLAGS \
    --checkpoint "$CURRENT_CKPT" \
    --prune-type "$PRUNE_TYPE" \
    --amount "$AMOUNT" \
    --finetune-epochs "$PER_STEP_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --optimizer "$OPTIMIZER" \
    --lr "$LR" \
    --scheduler "$SCHEDULER" \
    --t-max "$T_MAX" \
    --data-dir "$DATA_DIR" \
    --output-checkpoint "$OUTPUT_CKPT" \
    $PRUNE_FLAGS \
    $EXTRA_FLAGS \
    | tee "$OUT_DIR/step_$(printf "%02d" $STEP_IDX)_${PRUNE_TYPE}_amount${AMOUNT}_prune.json"

  CURRENT_CKPT="$OUTPUT_CKPT"

  python -u -m pruning_lab.main test \
    --model "$MODEL" \
    $MODEL_IMG_FLAGS \
    --checkpoint "$CURRENT_CKPT" \
    --batch-size 256 \
    --data-dir "$DATA_DIR" \
    | tee "$OUT_DIR/step_$(printf "%02d" $STEP_IDX)_${PRUNE_TYPE}_amount${AMOUNT}_test.json"

  STEP_IDX=$((STEP_IDX + 1))
done

echo "\nAll iterations complete. Final checkpoint: $CURRENT_CKPT"

# Produce a compact summary JSON using Python
python - << 'PY'
import json, os, re, glob
out_dir = os.environ.get('OUT_DIR', '') or 'runs'
pattern_test = os.path.join(out_dir, 'step_*_test.json')
pattern_prune = os.path.join(out_dir, 'step_*_prune.json')
tests = sorted(glob.glob(pattern_test))
prunes = sorted(glob.glob(pattern_prune))
def _read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
baseline = _read(os.path.join(out_dir, 'step_00_baseline_test.json'))
steps = []
for t in tests:
    m = re.search(r'step_(\d+)_([a-z]+)_amount([0-9.]+)_test.json', os.path.basename(t))
    if not m:
        # baseline
        continue
    idx, ptype, amount = int(m.group(1)), m.group(2), float(m.group(3))
    prune_path = t.replace('_test.json', '_prune.json')
    prune_obj = _read(prune_path)
    test_obj = _read(t)
    steps.append({
        'step': idx,
        'prune_type': ptype,
        'amount': amount,
        'global_sparsity': prune_obj.get('global_sparsity') if prune_obj else None,
        'checkpoint': prune_obj.get('checkpoint') if prune_obj else None,
        'accuracy': test_obj.get('accuracy') if test_obj else None,
        'loss': test_obj.get('loss') if test_obj else None,
    })
summary = {
    'model': os.environ.get('MODEL'),
    'prune_type': os.environ.get('PRUNE_TYPE'),
    'baseline': baseline,
    'steps': steps,
}
with open(os.path.join(out_dir, 'iterative_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(json.dumps({'summary_path': os.path.join(out_dir, 'iterative_summary.json')}, indent=2))
PY


