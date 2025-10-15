# Knowledge Distillation Lab - Quick Start Guide

## 🚀 Fastest Way to Complete the Lab

### On the Cluster (Recommended)

```bash
# 1. Submit all training jobs at once (they'll queue automatically)
cd /home/jxl2244/ecse397-efficient-deep-learning

# Train teachers (prerequisite for distillation)
sbatch -C gpu2h100 distillation_lab/utils/train_cnn_teacher.slurm
sbatch -C gpu2h100 distillation_lab/utils/train_vit_teacher.slurm
sbatch -w classt25 distillation_lab/utils/train_vit_teacher.slurm

# Train student baselines (can run in parallel with teachers)
sbatch -C gpu2h100 distillation_lab/utils/train_cnn_student_baseline.slurm
sbatch -C gpu2h100 distillation_lab/utils/train_vit_student_baseline.slurm

# Note: Wait for teachers to complete before submitting distillation jobs
# Check: ls distillation_lab/models_saved/cnn_teacher.pth

# Once teachers are done, submit distillation jobs
sbatch -C gpu2h100 distillation_lab/utils/distill_cnn.slurm
sbatch -C gpu2h100 distillation_lab/utils/distill_vit.slurm

# 2. Monitor jobs
squeue -u $USER
tail -f logs/distill_*.out

# 3. After all jobs complete, generate report (CPU OK)
module purge && module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
python3 -m distillation_lab.main report

# 4. Verify deliverables
ls distillation_lab/models_saved/
cat distillation_lab/report.json

# 5. Submit
zip -r jxl2244_distillation_lab.zip distillation_lab/
```

## 📋 Expected Timeline

| Phase | Jobs | Time | Can Parallelize? |
|-------|------|------|-----------------|
| Teachers | 2 | 24h each | ✅ Yes |
| Student Baselines | 2 | 16h each | ✅ Yes |
| Distillation | 2 | 16h each | ✅ Yes (after teachers) |

**Total time: ~3-4 days** (if teachers trained first, then baselines + distillation in parallel)

## 📦 Deliverables Checklist

After all jobs complete, verify:

```bash
# Check all 6 checkpoints exist
ls -lh distillation_lab/models_saved/*.pth
```

Should show:
- ✅ `cnn_teacher.pth`
- ✅ `cnn_student_no_kd.pth`
- ✅ `cnn_student_with_kd.pth`
- ✅ `vit_teacher.pth`
- ✅ `vit_student_no_kd.pth`
- ✅ `vit_student_with_kd.pth`

And:
- ✅ `report.json` (auto-generated)

## 🔍 Monitoring Commands

```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job <job_id>

# View output logs
tail -f logs/distill_cnn_teacher_*.out

# Check GPU usage (from within a job)
nvidia-smi

# Check completed jobs
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS
```

## ⚡ Quick Test (Optional)

Test locally before submitting to cluster:

```bash
cd /home/jxl2244/ecse397-efficient-deep-learning
source .venv/bin/activate

# Test imports and model creation
python3 distillation_lab/test_setup.py

# Quick 1-epoch test (if you have GPU access locally)
python3 -m distillation_lab.main train-teacher \
  --model resnet18 \
  --epochs 1 \
  --batch-size 32
```

## 🛠️ Troubleshooting

### "Teacher checkpoint not found"
```bash
# Check if teacher training completed
ls distillation_lab/models_saved/cnn_teacher.pth
# If missing, wait for teacher job or check logs
tail logs/distill_cnn_teacher_*.err
```

### "CUDA out of memory"
Edit SLURM scripts to reduce batch size:
```bash
# In the .slurm file, change:
BATCH_SIZE=${BATCH_SIZE:-64}  # was 128
```

### "Job still pending"
```bash
# Check queue status
squeue -p markov_gpu
# Try different GPU if needed
sbatch -C gpu2080 distillation_lab/utils/train_cnn_teacher.slurm
```

## 📊 Expected Results

Your `report.json` should look like:

```json
{
    "cnn": {
        "teacher_accuracy": 0.91-0.93,
        "student_accuracy_without_kd": 0.84-0.87,
        "student_accuracy_with_kd": 0.87-0.89
    },
    "vit": {
        "teacher_accuracy": 0.92-0.94,
        "student_accuracy_without_kd": 0.81-0.85,
        "student_accuracy_with_kd": 0.85-0.88
    }
}
```

**Key observation:** Students with KD should be 2-4% better than baselines!

## 🎯 Submission

```bash
# 1. Verify all files present
tree distillation_lab -I "__pycache__|*.pyc"

# 2. Create submission zip
zip -r jxl2244_distillation_lab.zip distillation_lab/ \
  -x "distillation_lab/__pycache__/*" \
  -x "distillation_lab/*/__pycache__/*"
  -x "distillation_lab/IMPLEMENTATION_SUMMARY.md"
  -x "distillation_lab/Lab-2.md"
  -x "distillation_lab/README.md"
  -x "distillation_lab/test_setup.py"

# 3. Verify zip contents
unzip -l jxl2244_distillation_lab.zip | head -20

# 4. Submit to Canvas before October 15, 11:59 PM EST
```

## 📖 For More Details

- `README.md` - Comprehensive usage guide
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `Lab-2.md` - Original assignment requirements

---

**Good luck! 🎓**

