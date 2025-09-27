# Synopsis

You will connect to the academic cluster Markov to run computations as part of your coursework. Below we summarize different methods for establishing a connection to the cluster, and for running computations. The methods you need to use will depend on the course objectives and the materials that the instructor has prepared.

## Connecting to Markov

Before you can run computations, you will need to establish a connection, or "session" with the cluster that verifies your identity.

### Web Based (recommended)

This connection is provided by a project called Open OnDemand and will let you launch terminals, applications like Jupyter or RStudio and even full desktop environments. You will connect to https://ondemand-markov.case.edu and authenticate with your network ID and password.

The site does not require the use of VPN and should be reachable from campus wireless and from off campus. 

The site is integrated with DUO and you may need to confirm the connection through one of the available DUO methods.

### SSH

This connection is text based without additional setup. You can connect directly to a login node using your operating systems ssh client. Linux and Mac OS both include ssh clients. For Windows, we generally recommend the use of the PuTTY ssh client[1], though newer versions of Windows do have a native client available[2].

The server name you will use is markov.case.edu, e.g.
```bash
    ssh markov.case.edu
```
This method requires the use of either a hardwired campus connection or the VPN.

Graphical applications will not work unless you establish X forwarding. Additional instructions for X forwarding are available here.

## Running Computations

Depending on the course you are taking, you may have a significantly customized environment for running your computations, such as custom applications in OnDemand. In this case the course materials will include the directions for how to launch the applications. If your course is launching computations directly on the cluster using the command line tools, the following commands will be useful.

### Terminology

- **Job** - A computation you want to run on the cluster
- **Queue** - The list of jobs running and waiting to run on the cluster
- **Scheduler** - The software the manages jobs and the queue
- **Compute node** - A computer designated to running computational tasks
- **Login Node** - A node designated for managing user connections and submitting jobs

### Where to Run Commands

If you have connected through the recommended OnDemand method above, you will see an option at the top "Clusters", under which you would click the option "_markov shell". This will start a terminal on one of the login nodes, e.g. hpc-login, hpc-login2 or markov01.

If you connected directly through ssh, your terminal will already have a shell running on one of the login nodes, e.g. hpc-login, hpc-login2 or markov01.

**DO NOT** run computational tasks directly on the login nodes. These nodes are where you will run the job management commands below to submit new jobs to the cluster.

### Start a Batch Job

Batch jobs run on the cluster without needing user input. They are the most efficient way to run computations, and part of this is because they are well defined. All steps of the computation are included in a script which is then submitted to the scheduler using the `sbatch` command.

```bash
    #!/bin/bash

    #SBATCH -c 2           # 2 CPUs   

    #SBATCH -A <PI> -p markov_cpu # PI account and partition

    #SBATCH --mem=8G       # 8 GB of RAM 

    #SBATCH --time=1       # Runtime of 1 minutes

    sleep 30
```

For GPU partion, use -p markov_gpu. Check Markov resource view for details.

Submit command:

```bash
    [stm@class-login ~]$ sbatch -p markov_cpu myscript.slurm 
    Submitted batch job 19169393
```

### Start an Interactive Job

Interactive jobs are good for debugging. They establish a terminal running on the computational resources so that you can test commands and scripts interactively, much the same as if you were testing on your own local computer.

```bash
[stm@class-login ~]$ salloc -A sxg125_csds438 -p markov_cpu --time=30 -c 2 --mem=8G srun --pty /bin/bash

salloc: Granted job allocation 19169394

salloc: Nodes classt01 are ready for job

[stm@classt01 ~]$ exit

exit

salloc: Relinquishing job allocation 19169394

salloc: Job allocation 19169394 has been revoked.

[stm@class-login ~]$ 
```

Note how the prompt changed from stm@class-login to stm@classt01. This reflects that your commands are running on the compute node rather than the login node. Typing exit or Ctrl+D will end the interactive job and the prompt will return to the login node.

### View My Jobs

You list you jobs in the queue with the squeue command with the --me option:

```bash
    [stm@class-login ~]$ squeue --me

    JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)

    19169392    markov_cpu sys-dash      stm  R       0:17      1 classct003
```

### Cancel My Job

You will use the scancel command with the job id as an argument:

```bash
    [stm@class-login ~]$ scancel 19169392
```

### Using Software Modules
Please review Module System Basics and Module System In-depth. 



# Module System Basics

## Module Load and Unload

Simply stated, a module is a packaging of definitions of environment variables into a script. In general, there is a module defined for each application, and that module defines the environment appropriately for that application. To use the application, you need to load the module first using the module command:

```bash
    module load <software-module>
```

For an example, if the software-module is "matlab", it becomes "module load matlab".

If there are more versions, let's say matlab/R2019b, then it becomes:

```bash

module load matlab/R2019b

```

To load the sotware-module, use the command:

```bash
module load <software-module>
```

To unload it, use:

```bash
module unload <software-module>
```

## Module List

To see which modules you have loaded, use:

```bash
    module list
```

### Viewing Modules That Can Be Loaded

Modules are organized into a hierarchy so that some modules can only be loaded after other modules have been loaded. To see the modules that can be loaded in the current environment use the command:

```bash
module avail
```

If you want to check the versions of a particular software such as matlab, then type:

```bash
module avail matlab
```

output:

```bash
...

matlab/R2018b          matlab/R2019b(default)
```

### Searching All Modules

To search all modules and determine what additional modules are required to use a specific software, you can use the "module spider <name>" command.  For example, R has dependency modules that must be loaded before R itself. First list available versions:

module spider R

Output:

-------------------------------------------------------------------------

  R:

-------------------------------------------------------------------------

    Description:

      R is a free software environment for statistical computing and graphics.



     Versions:

        R/3.2.5

        R/3.3.3

        R/3.4.2

        R/3.5.0

        R/3.5.3

        R/3.6.2

        R/4.0.2

Then select a specific version and use the command on the specific version:

module spider R/4.0.2

Output:

-----------------------------------------------------------------------------

  R: R/4.0.2

-----------------------------------------------------------------------------

    Description:

      R is a free software environment for statistical computing and graphics.



    You will need to load all module(s) on any one of the lines below before the "R/4.0.2" module is available to load.



      gcc/6.3.0  openmpi/2.0.1

Now, you need to load the R/4.0.2 module as:

module load gcc/6.3.0 openmpi/2.0.1

module load R/4.0.2

Module Display
To get the version, path of executable and libraries, and other information about the software-module (e.g. matlab):

module display <software-module>

For matlab, the output looks like:

-------------------------------------------------------------------

execute{cmd="source /usr/local/matlab/matlab-prefdir.sh", modeA={"load"}}

whatis("Name: matlab")

whatis("Version R2019b")

whatis("Category: library")

whatis("Description: Matlab R2019b")

pushenv("MATLAB","/usr/local/matlab/R2019b")

pushenv("MATLAB_COMM","native")

prepend_path("PATH","/usr/local/matlab/R2019b/bin")

prepend_path("LD_LIBRARY_PATH","/usr/local/matlab/R2019b/runtime/glnxa64")

...

-------------------------------------------------------------------

Here, the binary path for Matlab version 2019B is /usr/local/matlab/R2019b/bin


Module System In-Depth
We rely primarily on EasyBuild to install software for the HPC computing clusters, and implement module naming such that; 1)  all module files are directly available for loading; and, 2)  each module name uniquely identifies a particular installation.


Since all the modules are available to load at all times, it is important to ensure that the environment remains consistent, and the module command remains the best means to do so.


Toolchains
Toolchains are used to provide software built from the same 'base' package versions, and therefore benefitting from the associated consistency and community testing. New toolchains are released approximately twice each year. Case HPC will maintain the 'foss' and 'intel' toolchains.

The foss toolchain consists of all open source components (hence the name: "FOSS" stands for Free & Open Source Software): GCC, Open MPI, OpenBLAS, ScaLAPACK and FFTW.

The intel toolchain consists of the Intel C, C++ and Fortran compilers (on top of a GCC version controlled through EasyBuild) alongside the Intel MPI and Intel MKL libraries.



Lmod and the 'flat' hierarchy
We install our software as modules and we manage them using Lmod.

"Lmod is a Lua based module system that easily handles the MODULEPATH Hierarchical problem. Environment Modules provide a convenient way to dynamically change the users' environment through modulefiles. This includes easily adding or removing directories to the PATH environment variable. Modulefiles for Library packages provide environment variables that specify where the library and header files can be found." Source: TACC Lmod


Module Paths
Modules installed with EasyBuild will have a common module path: /usr/local/easybuild/modules/all

The MODULEPATH variable is set by default to reference the EasyBuild path. Custom software, specifcially that installed from source in a personal directory (e.g. /home/<caseid>/..) may have Lmod modules established. The MODULEPATH will require modification to make those modules available to use for managing the shell environment for this custom software. 


Optimization Trees
The CWRU HPC has nodes with a variety of capabilities, including the cpu instructions sets. To manage code installed to run on either a small set of specialized nodes, or to run generally on a broad set of nodes, we are using using module trees. The current trees and their capabilities are listed:

allnodes: (default)  should run without error on all the HPC nodes, though without the latest cpu architecture support.

icosa:      optimized to use all cpu architecture support on the icosa192gb feature nodes.

aisc:        optimized for use with the aisc partition nodes (limited access).


To switch trees, there are two approaches. Both adjust the MODULEPATH environment variable.

Approach 1

export MODULEPATH=/usr/local/easybuild_<tree>/modules/all

Approach 2

module unuse /usr/local/easybuild_allnodes/modules/all

module use /usr/local/easybuild_<tree>/modules all

The <tree> is a placeholder, with the target to be taken from the list of trees above (e.g. easybuild_allnodes)


Supplementary Software
The latest versions of licensed software and other system software supplementing that available from EasyBuild will be installed in another location, from the file system root location "/usr/local/software". For example, Matlab will be found under "/usr/local/software/matlab/<release-version>". This path is also included by default in MODULEPATH.

Managing Modules
Module Load and Unload
Simply stated, a module is a packaging of definitions of environment variables into a script. In general, there is a module defined for each application, and that module defines the environment appropriately for that application. To use the application, you need to load the module first using the module command:

module load <software-module>

For an example, if the software-module is "matlab", it becomes "module load matlab".

If there are more versions, let's say matlab/R2023a, then it becomes:

module load matlab/R2023a

To load the sotware-module, use the command:

module load <software-module>

To unload it, use:

module unload <software-module>

Module List
To see which modules you have loaded, use:

module list

All Modules Can Be Loaded - Manage environment with care
Contrary to previous clusters, Modules on Pioneer are organized without hierarchy, and the module names contain information about toolchains and compilers used in the build. For example, 

module avail Python/   (note the trailing '/')

[mrd20@hpc8 ~]$ module avail Python/   #  partial output

------------------------------------- /usr/local/easybuild_allnodes/modules/all -----------------------

   GitPython/3.1.40-GCCcore-12.3.0      Python/3.10.4-GCCcore-11.3.0

   LIBSVM-Python/3.30-foss-2022a        Python/3.10.8-GCCcore-12.2.0-bare

   Python/3.7.4-GCCcore-8.3.0           Python/3.10.8-GCCcore-12.2.0

   Python/3.8.2-GCCcore-9.3.0           Python/3.11.3-GCCcore-12.3.0

   Python/3.8.6-GCCcore-10.2.0          Python/3.11.5-GCCcore-13.2.0

   Python/3.9.5-GCCcore-10.3.0-bare     Python/3.12.3-GCCcore-13.3.0          (D)


All modules with 'python/' or 'Python/' are listed. This example shows the distinction between the GCCcore prepared modules, and the 'gompic' modules. Without the trailing '/', modules having "python" appearing anywhere in the name will be listed. 


Possible Compatibility Issues

For compatibility reasons, try to load the dependency modules built with appropriate toolchains and versions. For example, the dependency modules are using the same GCCcore-12.2.0.

module purge                                                # purge all the modules before loading any new modules

module load Python/3.10.8-GCCcore-12.2.0

module load Tk/8.6.12-GCCcore-12.2.0

module load Tcl/8.6.12-GCCcore-12.2.0


Let's consider a scenario where you want to use PyYAML module. Let's say you decide to use the version "PyYAML/6.0-GCCcore-12.3.0". Then you need to check the possible dependency modules (using "module list" after loading the module). Now, you will be using "Python/3.11.3-GCCcore-12.3.0" version of Python. The other version of Python can have compatibility issues. Now, if you want to use SciPy module, you will be choosing "module show SciPy-bundle/2023.07-gfbf-2023a" that uses the same Python version.

Searching All Modules
The distinction between using 'avail' and 'spider' is now reduced, as all modules will be listed, according to what directories are included in the MODULEPATH variable (echo $MODULEPATH).  The two commands provide different degrees of information, with 'module avail' providing a listing of module names, and "module spider <name>" the set of modules, a description of the software, and further steps to gain more information. Following from the previous example, use "module spider gompi/' and "module spider gompic/" to better understand 'gompic':


[mrd20@hpc8 ~]$ module spider gompi/


	Description:  GNU Compiler Collection (GCC) based compiler toolchain, including OpenMPI for MPI support.

 	Versions:    	  gompi/2020a, gompi/2020b, gompi/2021a, gompi/2021b, gompi/2022a

                              gompi/2022b, gompi/2023a, gompi/2023b, gompi/2024a


As previously, for detailed information about a specific package (including how to load the modules) use the module's full name.   For example:


 	$ module spider gompi/2021a


Note that with the 'flat hierarchy' and extended module names, tab-completion of module names is available in the shell.

