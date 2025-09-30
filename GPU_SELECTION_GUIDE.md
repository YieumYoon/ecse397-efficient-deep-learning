# GPU Selection Guide for Markov Cluster

## 🎮 Available GPUs (Performance Ranking)

| GPU Type | Model | VRAM | Nodes | Performance | Best For |
|----------|-------|------|-------|-------------|----------|
| `gpu2h100` | NVIDIA H100 | ~80GB | 2 | ⭐⭐⭐⭐⭐ Best | Large models, fastest training |
| `gpu4090` | NVIDIA RTX 4090 | 24GB | 1 | ⭐⭐⭐⭐ Excellent | High-performance training |
| `gpul40s` | NVIDIA L40S | 48GB | 2 | ⭐⭐⭐⭐ Excellent | Large batches, good VRAM |
| `gpu4070` | NVIDIA RTX 4070 | 12GB | 1 | ⭐⭐⭐ Good | Standard training |
| `gpu2080` | RTX 2080 Ti | 11GB | 14+ | ⭐⭐⭐ Good | Most available, reliable |

---

## 🔍 Checking GPU Availability

### Quick Check
```bash
si  # Shows all GPU nodes and their status
```

### Check Specific GPU Type
```bash
si | grep gpu2h100    # Check H100 availability
si | grep gpu4090     # Check RTX 4090 availability
si | grep gpu2080     # Check RTX 2080 Ti availability
```

### Understanding the Output
```
PARTITION  AVAIL_FEATURES  STATE  CPUS(A/I/O/T)  NODELIST
markov_gpu    gpu2h100      idle   0/72/0/72      classt[23-24]
```

- **STATE**:
  - `idle` - Fully available ✅
  - `mix` - Partially used (some CPUs free) ⚠️
  - `alloc` - Fully allocated ❌
  - `down` - Offline ❌
  - `drain` - Being drained ❌

- **CPUS(A/I/O/T)**:
  - A = Allocated (in use)
  - I = Idle (available)
  - O = Other
  - T = Total

---

## 🚀 Method 1: Automatic GPU Selection (Recommended)

### Use the Helper Script

The automatic selector chooses the best available GPU in this order:
1. H100 (best performance)
2. RTX 4090
3. L40S
4. RTX 4070
5. RTX 2080 Ti (most available)

```bash
# Submit with auto-selected best GPU
bash scripts_corrected/submit_best_gpu.sh scripts_corrected/train_cnn.slurm

# With custom parameters
EPOCHS=100 bash scripts_corrected/submit_best_gpu.sh scripts_corrected/train_cnn.slurm
```

**Output Example:**
```
Checking GPU availability...
Selected: gpu2h100

Current GPU status:
markov_gpu    gpu2h100    idle   0/72/0/72    classt[23-24]

Submitting job: scripts_corrected/train_cnn.slurm
GPU constraint: -C gpu2h100
```

---

## 🎯 Method 2: Manual GPU Selection

### Select Specific GPU Type

```bash
# Request H100 (best, but limited to 2 nodes)
sbatch -C gpu2h100 scripts_corrected/train_cnn.slurm

# Request RTX 4090 (excellent performance)
sbatch -C gpu4090 scripts_corrected/train_cnn.slurm

# Request RTX 2080 Ti (most available)
sbatch -C gpu2080 scripts_corrected/train_cnn.slurm
```

### Add Constraint to SLURM Script

Add this line to your `.slurm` file:
```bash
#SBATCH -C gpu2h100    # or gpu4090, gpu2080, etc.
```

---

## 📊 Real-World Performance Comparison

### Training ResNet-18 on CIFAR-10 (300 epochs)

| GPU | Time | Speed vs 2080 Ti | Availability |
|-----|------|------------------|--------------|
| H100 | ~14-15h | 1.3-1.4x faster ⚡⚡ | 2 nodes |
| RTX 4090 | ~16-17h | 1.2x faster ⚡ | 1 node |
| L40S | ~17-18h | 1.1x faster | 2 nodes |
| RTX 2080 Ti | ~18-20h | Baseline | 14+ nodes |

*Note: Times are approximate and depend on batch size and other factors*

### Memory Comparison (Max Batch Size for ResNet-18)

| GPU | VRAM | Max Batch Size* |
|-----|------|----------------|
| H100 | 80GB | 2048+ |
| L40S | 48GB | 1024+ |
| RTX 4090 | 24GB | 512 |
| RTX 2080 Ti | 11GB | 256 |
| RTX 4070 | 12GB | 256 |

*Approximate values for ResNet-18 with FP16 mixed precision

---

## 💡 Best Practices

### When to Use Which GPU

**Use H100 when:**
- ✅ Training very large models
- ✅ Need fastest possible training
- ✅ Experimenting with large batch sizes
- ✅ Working with large datasets
- ⚠️ Only 2 nodes - may have wait time

**Use RTX 4090 when:**
- ✅ Need excellent performance
- ✅ H100 is busy
- ✅ Model fits in 24GB VRAM

**Use RTX 2080 Ti when:**
- ✅ Quick prototyping
- ✅ Standard training workloads
- ✅ Need immediate availability (14+ nodes)
- ✅ Model fits in 11GB VRAM
- ✅ Most reliable option

**Use L40S when:**
- ✅ Need more VRAM (48GB)
- ✅ Large batch sizes
- ✅ Middle ground between H100 and 2080 Ti

---

## 🛠️ Usage Examples

### Example 1: Auto-Select Best Available
```bash
# Let the script choose the best GPU
bash scripts_corrected/submit_best_gpu.sh scripts_corrected/train_cnn.slurm
```

### Example 2: Check Then Submit
```bash
# Check what's available
si | grep markov_gpu | grep idle

# Manually choose based on availability
sbatch -C gpu2h100 scripts_corrected/train_cnn.slurm
```

### Example 3: Try Best, Fallback to Available
```bash
# Try H100 first
sbatch -C gpu2h100 scripts_corrected/train_cnn.slurm

# If it's queued too long, cancel and use 2080 Ti
scancel <job_id>
sbatch -C gpu2080 scripts_corrected/train_cnn.slurm
```

### Example 4: In the SLURM Script
```bash
#!/bin/bash
#SBATCH --job-name=my_training
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH -C gpu2h100          # Request H100
#SBATCH --cpus-per-task=16   # H100 nodes have more CPUs
#SBATCH --mem=64G            # Can use more memory
...
```

---

## 📝 Integration with Existing Scripts

### Option 1: Use Wrapper Script (Easiest)
```bash
bash scripts_corrected/submit_best_gpu.sh scripts_corrected/train_cnn.slurm
bash scripts_corrected/submit_best_gpu.sh scripts_corrected/train_vit.slurm
bash scripts_corrected/submit_best_gpu.sh scripts_corrected/prune_cnn.slurm
```

### Option 2: Manual Check and Submit
```bash
# Check availability
GPU_TYPE=$(bash scripts_corrected/select_best_gpu.sh)
echo "Best GPU: $GPU_TYPE"

# Submit with selected GPU
sbatch -C $GPU_TYPE scripts_corrected/train_cnn.slurm
```

### Option 3: Add to Script Header
Edit your `.slurm` file to add:
```bash
#SBATCH -C gpu2h100  # or run: select_best_gpu.sh to choose
```

---

## 🔧 Advanced: Custom GPU Selection Logic

### Create Your Own Selector

```bash
#!/bin/bash
# custom_gpu_select.sh

# Get GPU status
GPU_STATUS=$(si | grep markov_gpu)

# Example: Only use H100 or 4090, never queue for 2080 Ti
if echo "$GPU_STATUS" | grep -q "gpu2h100.*idle"; then
    echo "gpu2h100"
elif echo "$GPU_STATUS" | grep -q "gpu4090.*idle"; then
    echo "gpu4090"
else
    echo "No preferred GPU available, waiting..."
    exit 1
fi
```

---

## 📊 Current Cluster Status (as of test)

```
GPU Type    | Idle Nodes | Status
------------|------------|------------------
gpu2h100    | 2          | ✅ Available
gpu4090     | 0          | ⚠️ Mix (some CPUs free)
gpul40s     | 0          | ⚠️ Mix/Drain
gpu4070     | 1          | ✅ Available
gpu2080     | 14         | ✅✅✅ Highly Available
```

Run `si` to see current real-time status.

---

## 🎓 Tips and Tricks

### Maximize H100 Usage
```bash
# H100 has 96 CPUs per node - use more workers
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
...
--workers 16  # in your Python command
```

### Quick Availability Check Script
```bash
# Add to ~/.bashrc or ~/.bash_profile
alias gpu_status='si | grep markov_gpu | grep -E "idle|mix"'
alias gpu_h100='si | grep gpu2h100'
alias gpu_best='bash ~/ecse397-efficient-deep-learning/scripts_corrected/select_best_gpu.sh'
```

Then use:
```bash
gpu_status  # Quick check
gpu_h100    # Check H100 specifically
gpu_best    # Get best available GPU
```

---

## 📚 Summary

### Quick Reference Commands

```bash
# Check all GPUs
si

# Auto-submit to best GPU
bash scripts_corrected/submit_best_gpu.sh <script.slurm>

# Submit to specific GPU
sbatch -C gpu2h100 <script.slurm>  # H100
sbatch -C gpu4090 <script.slurm>   # RTX 4090
sbatch -C gpu2080 <script.slurm>   # RTX 2080 Ti

# Get best GPU programmatically
GPU=$(bash scripts_corrected/select_best_gpu.sh)
```

### Decision Tree

```
Need fastest? → H100 (if available)
    ↓ busy
Need high perf? → RTX 4090 (if available)
    ↓ busy
Need big VRAM? → L40S (if available)
    ↓ busy
Need reliable? → RTX 2080 Ti (14+ nodes, almost always available)
```

---

**The automatic selector handles this for you!**  
Just run: `bash scripts_corrected/submit_best_gpu.sh <your_script.slurm>`

---

*Last updated: September 30, 2025*  
*Run `si` for real-time status*
