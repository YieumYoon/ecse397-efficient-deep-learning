Here’s the comprehensive guide for using the CWRU Markov HPC cluster, formatted in Markdown for easy reference and sharing:

***

# CWRU Markov HPC Cluster Guide

## 1. Access & Login

- **Account Requirement:**  
  You must have an HPC account ([Access Policy](https://sites.google.com/a/case.edu/hpcc/hpc-cluster/quick-start/access-policy)).

- **Login Nodes for Coursework:**  
  - **SSH (Linux/Mac):**
    ```bash
    ssh -X <CaseID>@markov.case.edu
    ```
  - **Web Portal:**  
    [Markov OnDemand Portal](https://ondemand-markov.case.edu/) *(no VPN; DUO required)*
  - **Windows:**  
    [X2Go](https://sites.google.com/a/case.edu/hpcc/hpc-cluster/hpc-visual-access/x2go) or [MobaXterm](https://sites.google.com/a/case.edu/hpcc/hpc-cluster/hpc-visual-access/mobaxterm)

***

## 2. Basic Rules and Good Practice

- **Never run jobs directly on the login node.**  
  Use `sbatch` (batch jobs) or request interactive nodes with `srun`/`salloc`.
- **Keep each major job in its own working directory.**
- **Interactive sessions may disconnect after several hours;** use batch jobs for longer tasks.

***

## 3. Submitting Jobs

- **Batch Submission (recommended):**

    ```bash
    #!/bin/bash
    #SBATCH -c 4
    #SBATCH --mem=8g
    #SBATCH -p markov_cpu
    #SBATCH --time=02:00:00
    #SBATCH --job-name=test_job
    module load matlab
    matlab -nodisplay -r "your_code"
    ```
    Submit with:
    ```bash
    sbatch my_job.sh
    ```

- **Interactive Job:**

    ```bash
    salloc -c 4 --mem=8g -p markov_cpu srun --pty /bin/bash
    ```

***

## 4. SLURM Commands Cheat Sheet

```bash
# View your jobs
squeue -u <CaseID>

# Cancel a job
scancel <JOBID>

# Cancel all your jobs
scancel -u <CaseID>

# See node state
scontrol show job <JOBID>

# System/node usage
top

# Quick summary (if available)
i
```

***

## 5. Using Modules

```bash
# Load a module
module load <modulename>

# Unload a module
module unload <modulename>

# List loaded modules
module list

# See what modules are available
module avail

# Search for software/dependencies
module spider <modulename>

# Display module info (paths, versions)
module display <modulename>
```

***

## 6. Storage and Quotas

- **Home Directory:** `/home` (member ~1.2TB, guest ~300GB)
- **Scratch/Temp:** `/scratch`, `$PFSDIR`, `/mnt/fs1`
- **Check quotas:**

    ```bash
    quotagrp
    # or
    quotagrp2
    ```

***

## 7. File Transfer

- **Internal:** Use the hpctransfer node.
- **External:** Use Globus, `rsync`, `scp`, WinSCP, FileZilla, etc.
- **Portal:** [File Transfer Reference](https://sites.google.com/a/case.edu/hpcc/data-transfer)

***

## 8. Monitoring, Output, Debugging

- **Job Status & Output:**

    ```bash
    squeue
    sstat
    cat <outputfile>
    tail -f <outputfile>
    ```

- **Notifications:** *(add to job script for email results)*

    ```bash
    #SBATCH --mail-user=<your_email>
    #SBATCH --mail-type=END,FAIL
    ```

***

## 9. Troubleshooting & Common Issues

- **Pending Jobs:**
    - Resource request too large, wall time exceeded, group allocation maxed out.
    - Check with:
      ```bash
      scontrol show job <JOBID>
      ```
    - Contact support if unresolved.

- **Memory errors:**  
    - Request more memory with `--mem=XXg`.

- **Environment/path issues:**  
    - Reset `.bashrc` from `/etc/skel/.bashrc` if OnDemand or jobs fail.

- **Reference:**  
    [Running Jobs FAQ](https://sites.google.com/a/case.edu/hpcc/guides-and-training/faqs/running-jobs)

***

## 10. Help & Support

- **Documentation:**  
  [User Guide](https://sites.google.com/a/case.edu/hpcc/hpc-cluster/quick-start), [FAQ](https://sites.google.com/a/case.edu/hpcc/guides-and-training/faqs/running-jobs)
- **Email:** hpc-supportATcase.edu
- **Office Hours:** Wed 2–4pm (email for link)

***

## Sample Workflow: Submitting a MATLAB Batch Job

```bash
# Login
ssh -X <CaseID>@markov.case.edu
# Prepare job script (my_job.sh), example above
# Submit job
sbatch my_job.sh
# Monitor
squeue -u <CaseID>
# Check output
cat slurm-<jobid>.out
```

***

This guide includes all essential instructions, etiquette, commands, and troubleshooting tips for student use of the CWRU Markov HPC cluster. Reach out to support or reference documentation for specific needs!

[1](https://sites.google.com/a/case.edu/hpcc/home)