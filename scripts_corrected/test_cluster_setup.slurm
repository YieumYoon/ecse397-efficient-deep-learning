#!/bin/bash
#SBATCH --job-name=test_cluster
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/jxl2244/ecse397-efficient-deep-learning/logs/test_cluster_%j.out
#SBATCH --error=/home/jxl2244/ecse397-efficient-deep-learning/logs/test_cluster_%j.err

# Cluster Setup Verification Script
# This script tests that the cluster environment is properly configured

set -euo pipefail

echo "======================================"
echo "MARKOV CLUSTER SETUP VERIFICATION"
echo "======================================"
echo ""

# Test 1: Job Information
echo "Test 1: Job Information"
echo "-----------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# Test 2: Scratch Space Access
echo "Test 2: Scratch Space Access"
echo "-----------------------------"
echo "TMPDIR: $TMPDIR"
echo "PFSDIR: $PFSDIR"
WORK_DIR="$TMPDIR"
echo "Using work directory: $WORK_DIR"
cd "$WORK_DIR" && echo "✓ Successfully changed to work directory"
echo "Current directory: $(pwd)"
echo "Disk space available:"
df -h $WORK_DIR
echo ""

# Test 3: Write/Read Test
echo "Test 3: Scratch Space I/O"
echo "--------------------------"
echo "Test data" > test_file.txt && echo "✓ Successfully wrote test file"
cat test_file.txt > /dev/null && echo "✓ Successfully read test file"
rm test_file.txt && echo "✓ Successfully deleted test file"
echo ""

# Test 4: Module System
echo "Test 4: Module System"
echo "----------------------"
module purge && echo "✓ Module purge successful"
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 && echo "✓ PyTorch module loaded"
echo ""

# Test 5: Python Environment
echo "Test 5: Python Environment"
echo "---------------------------"
which python3 && echo "✓ Python3 found"
PYTHON_VERSION=$(python3 --version)
echo "Python version: $PYTHON_VERSION"
echo ""

# Test 6: PyTorch Installation
echo "Test 6: PyTorch Installation"
echo "-----------------------------"
python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("✗ CUDA not available!")
        sys.exit(1)
except ImportError as e:
    print(f"✗ Failed to import PyTorch: {e}")
    sys.exit(1)

try:
    import torchvision
    print(f"✓ TorchVision version: {torchvision.__version__}")
except ImportError:
    print("✗ TorchVision not available")
    sys.exit(1)

print("\n✓ All PyTorch checks passed!")
EOF

if [ $? -eq 0 ]; then
    echo "✓ PyTorch test successful"
else
    echo "✗ PyTorch test failed"
    exit 1
fi
echo ""

# Test 7: GPU Compute Test
echo "Test 7: GPU Compute Test"
echo "-------------------------"
python3 << 'EOF'
import torch

# Simple GPU computation test
try:
    # Create tensors on GPU
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    
    # Perform computation
    z = torch.matmul(x, y)
    
    # Check result
    assert z.device.type == 'cuda', "Result not on GPU"
    assert z.shape == (1000, 1000), "Result shape incorrect"
    
    print("✓ GPU computation test passed")
    print(f"  - Created 1000x1000 matrices on GPU")
    print(f"  - Performed matrix multiplication")
    print(f"  - Result verified successfully")
except Exception as e:
    print(f"✗ GPU computation test failed: {e}")
    import sys
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "✓ GPU test successful"
else
    echo "✗ GPU test failed"
    exit 1
fi
echo ""

# Test 8: Data Download Test (small dataset)
echo "Test 8: Data Download Test"
echo "---------------------------"
python3 << 'EOF'
import torch
from torchvision import datasets, transforms

try:
    # Try to download a small dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    # This will download to ./data in scratch space
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    print(f"✓ Successfully downloaded CIFAR-10 test set")
    print(f"  - Dataset size: {len(dataset)}")
    print(f"  - First sample shape: {dataset[0][0].shape}")
    
except Exception as e:
    print(f"✗ Data download failed: {e}")
    import sys
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "✓ Data download test successful"
else
    echo "✗ Data download test failed"
    exit 1
fi
echo ""

# Test 9: Copy Results Back
echo "Test 9: Copy Results Back to Home"
echo "-----------------------------------"
TEST_RESULT_FILE="cluster_test_results.txt"
echo "Cluster test completed successfully at $(date)" > "$TEST_RESULT_FILE"
echo "Job ID: $SLURM_JOB_ID" >> "$TEST_RESULT_FILE"
echo "Node: $SLURMD_NODENAME" >> "$TEST_RESULT_FILE"

mkdir -p $HOME/ecse397-efficient-deep-learning/logs
cp "$TEST_RESULT_FILE" $HOME/ecse397-efficient-deep-learning/logs/ && echo "✓ Successfully copied results to home"
echo ""

# Test 10: Cleanup
echo "Test 10: Scratch Space Cleanup"
echo "--------------------------------"
echo "✓ TMPDIR is automatically cleaned by SLURM after job completion"
echo "  No manual cleanup required"
echo ""

# Summary
echo "======================================"
echo "ALL TESTS PASSED! ✓"
echo "======================================"
echo ""
echo "Your cluster setup is working correctly!"
echo "You can now submit jobs using the corrected scripts."
echo ""
echo "Results saved to: $HOME/ecse397-efficient-deep-learning/logs/$TEST_RESULT_FILE"
echo ""
