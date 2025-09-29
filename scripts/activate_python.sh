#!/bin/bash
# Load Python environment for SLURM jobs

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

export PATH=$HOME/.local/bin:$PATH

echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"