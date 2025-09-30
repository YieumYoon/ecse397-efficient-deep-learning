#!/bin/bash
# Script to automatically select the best available GPU on Markov cluster
# Usage: GPU_TYPE=$(bash select_best_gpu.sh)

# Get GPU status from si command
GPU_STATUS=$(si 2>/dev/null | grep markov_gpu | grep -E 'idle|mix')

# GPU preference order (best to worst)
GPU_TYPES=("gpu2h100" "gpu4090" "gpul40s" "gpu4070" "gpu2080")

# Function to count idle nodes for a GPU type
count_idle_nodes() {
    local gpu_type=$1
    echo "$GPU_STATUS" | grep "$gpu_type" | grep "idle" | wc -l
}

# Function to count mix nodes with available CPUs for a GPU type
count_mix_nodes() {
    local gpu_type=$1
    echo "$GPU_STATUS" | grep "$gpu_type" | grep "mix" | wc -l
}

# Find the best available GPU
BEST_GPU=""
MAX_IDLE=0

for gpu in "${GPU_TYPES[@]}"; do
    idle_count=$(count_idle_nodes "$gpu")
    mix_count=$(count_mix_nodes "$gpu")
    
    if [ "$idle_count" -gt 0 ] || [ "$mix_count" -gt 0 ]; then
        if [ -z "$BEST_GPU" ]; then
            # First available GPU in preference order
            BEST_GPU="$gpu"
            MAX_IDLE=$idle_count
        fi
        # Always prefer H100 or 4090 if available
        if [ "$gpu" = "gpu2h100" ] || [ "$gpu" = "gpu4090" ]; then
            BEST_GPU="$gpu"
            break
        fi
    fi
done

# If we found a GPU, print it
if [ -n "$BEST_GPU" ]; then
    echo "$BEST_GPU"
    
    # Print availability info to stderr (won't interfere with capture)
    >&2 echo "Selected: $BEST_GPU"
    >&2 si | grep markov_gpu | grep "$BEST_GPU"
else
    # No GPU available, default to gpu2080 (most nodes)
    >&2 echo "No idle GPUs found, defaulting to gpu2080"
    echo "gpu2080"
fi
