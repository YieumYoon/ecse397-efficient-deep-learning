#!/usr/bin/env python3
"""Submit all improvement jobs to SLURM."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return 1, "", str(e)

def main():
    print("=" * 80)
    print("SUBMITTING ALL IMPROVEMENT JOBS")
    print("=" * 80)
    
    # Change to project directory
    project_dir = Path("/home/jxl2244/ecse397-efficient-deep-learning")
    scripts_dir = project_dir / "scripts"
    
    # Submit ResNet-18 retraining
    print("\n1. Submitting ResNet-18 retraining job (classt23)...")
    ret, out, err = run_command(f"cd {project_dir} && sbatch {scripts_dir}/retrain_resnet18.sh")
    if ret == 0 and out:
        job1 = out.split()[-1]  # Get job ID
        print(f"   ✓ Submitted: Job ID {job1}")
    else:
        print(f"   ✗ Failed: {err or 'No output'}")
        job1 = None
    
    # Submit CNN unstructured pruning (depends on job1)
    print("\n2. Submitting CNN unstructured pruning (classt24)...")
    if job1:
        ret, out, err = run_command(
            f"cd {project_dir} && sbatch --dependency=afterok:{job1} {scripts_dir}/reprune_cnn_unstructured.sh"
        )
    else:
        ret, out, err = run_command(f"cd {project_dir} && sbatch {scripts_dir}/reprune_cnn_unstructured.sh")
    
    if ret == 0 and out:
        job2 = out.split()[-1]
        dep_msg = f" (depends on {job1})" if job1 else ""
        print(f"   ✓ Submitted: Job ID {job2}{dep_msg}")
    else:
        print(f"   ✗ Failed: {err or 'No output'}")
        job2 = None
    
    # Submit ViT unstructured pruning (independent)
    print("\n3. Submitting ViT unstructured pruning (classt25)...")
    ret, out, err = run_command(f"cd {project_dir} && sbatch {scripts_dir}/reprune_vit_unstructured.sh")
    if ret == 0 and out:
        job3 = out.split()[-1]
        print(f"   ✓ Submitted: Job ID {job3}")
    else:
        print(f"   ✗ Failed: {err or 'No output'}")
        job3 = None
    
    # Submit ViT structured pruning (independent)
    print("\n4. Submitting ViT structured pruning (classt06)...")
    ret, out, err = run_command(f"cd {project_dir} && sbatch {scripts_dir}/reprune_vit_structured.sh")
    if ret == 0 and out:
        job4 = out.split()[-1]
        print(f"   ✓ Submitted: Job ID {job4}")
    else:
        print(f"   ✗ Failed: {err or 'No output'}")
        job4 = None
    
    print("\n" + "=" * 80)
    print("SUBMISSION SUMMARY")
    print("=" * 80)
    
    jobs = [
        ("ResNet-18 Training (classt23)", job1),
        ("CNN Unstructured (classt24)", job2),
        ("ViT Unstructured (classt25)", job3),
        ("ViT Structured (classt06)", job4)
    ]
    
    success_count = sum(1 for _, jid in jobs if jid)
    
    for name, jid in jobs:
        status = f"✓ Job ID: {jid}" if jid else "✗ Failed to submit"
        print(f"  {name}: {status}")
    
    print(f"\n  Successfully submitted: {success_count}/4 jobs")
    
    if success_count > 0:
        print("\n" + "=" * 80)
        print("MONITORING")
        print("=" * 80)
        print("\nCheck job status:")
        print("  squeue -u $USER")
        print("\nView output logs:")
        print("  tail -f resnet18_retrain_*.out")
        print("  tail -f cnn_unstructured_prune_*.out")
        print("  tail -f vit_unstructured_prune_*.out")
        print("  tail -f vit_structured_prune_*.out")
        print("\nAfter all jobs complete:")
        print("  python scripts/update_report.py")
        print("")
    
    # Save job IDs
    if any(jid for _, jid in jobs):
        job_file = project_dir / "job_ids.txt"
        with open(job_file, "w") as f:
            for name, jid in jobs:
                if jid:
                    varname = name.upper().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                    f.write(f"{varname}={jid}\n")
        print(f"Job IDs saved to: {job_file}")
        print("")
    
    return 0 if success_count == 4 else 1

if __name__ == "__main__":
    sys.exit(main())

