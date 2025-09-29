#!/bin/bash
# Simple setup: Load PyTorch from HPC modules, install timm

set -e

echo "Setting up Python environment..."

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo "✓ PyTorch $(python3 -c 'import torch; print(torch.__version__)')"
echo "✓ torchvision $(python3 -c 'import torchvision; print(torchvision.__version__)')"

echo ""
echo "Installing timm..."
pip install --user timm

echo ""
echo "✓ Setup complete!"
echo ""
echo "To use: module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1"