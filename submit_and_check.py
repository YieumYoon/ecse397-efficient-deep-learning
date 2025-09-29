#!/usr/bin/env python3
import subprocess
import time

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

# Submit jobs
jobs = []
print("Submitting jobs...")

scripts = [
    "scripts/retrain_resnet18.sh",
    "scripts/reprune_vit_unstructured.sh",
    "scripts/reprune_vit_structured.sh"
]

for script in scripts:
    print(f"Submitting {script}...")
    ret, out, err = run_cmd(f"sbatch {script}")
    if ret == 0 and out:
        job_id = out.strip().split()[-1]
        jobs.append((script, job_id))
        print(f"  Success: Job ID {job_id}")
    else:
        print(f"  Failed: {err}")

# Wait a moment for jobs to register
time.sleep(2)

# Check queue
print("\nChecking queue...")
ret, out, err = run_cmd("squeue -u jxl2244")
print(out)

# Write to file
with open("/home/jxl2244/ecse397-efficient-deep-learning/JOB_STATUS.txt", "w") as f:
    f.write("=== JOBS SUBMITTED ===\n")
    for script, job_id in jobs:
        f.write(f"{script}: {job_id}\n")
    f.write("\n=== QUEUE STATUS ===\n")
    f.write(out)
    f.write("\n")

print("\nStatus written to JOB_STATUS.txt")

