# TurboWorkflows

<img src="logo/turboworkflows_logo.jpg" width="70%">

![license](https://img.shields.io/github/license/kousuke-nakano/turboworkflows) ![release](https://img.shields.io/github/release/kousuke-nakano/turboworkflows/all.svg) ![fork](https://img.shields.io/github/forks/kousuke-nakano/turboworkflows?style=social) ![stars](https://img.shields.io/github/stars/kousuke-nakano/turboworkflows?style=social)

`TurboWorkflows` is a python package realizing high-throuput quantum Monte Carlo calculations with the open-source ab-initio quantum Monte Carlo code, `TurboRVB`.

`TurboRVB` software family is now composed of the 4 layered packages:

- `TurboWorkflows` (Workflows for realizing QMC high-throughput calculations)
- `TurboGenius` (Advanced python wrappers and command-line tools)
- `pyturbo` (Python-Fortran90 wrappers)
- `TurboRVB` (Quantum Monte Carlo kernel)

`TurboWorkflows` is the fourth layer package.

# Beta version
This is a **beta** version!!!! Contact the developers whenever you find bugs. Any suggestion is also welcome!

# Features of `TurboWorkflows`
`TurboGenius` [https://github.com/kousuke-nakano/turbogenius] does not internally have any functionality to manage job submissions/collections not to ruin its generality. Therefore, one should submit a python script directly to a cluster machine if one wants to run DFT and QMC jobs sequentially. `TurboRVB` commands launched by `TurboGenius` and `PyTurbo` can be specified through environmental variables such as TURBOGENIUS QMC COMMAND. For instance, if you set TURBOGENIUS_QMC_COMMAND='mpirun -np 64 turborvb-mpi.x', you can launch VMC, LRDMC jobs, etc... with 64 MPI processes on a cluster machine. This is a straightforward way to realize a python workflow based on TurboGenius.

`TurboWorkflows` provides a more sophisticated way to realize workflows by combining `TurboGenius` with an internal file/job managing package. In `TurboWorkflows`, each workflow class inherits the parent Workflow class with options useful for a QMC calculation. For instance, in the `VMC_workflow`, a user can specify a target accuracy (i.e., statistical error) of a VMC calculation. The `VMC_workflow` first submits an initial VMC run to a machine with the specified MPI and OpenMP processes to get a stochastic error bar per Monte Carlo step. Since the error bar is inversely proportional to the square root of the number of Monte Carlo samplings, the necessary steps to achieve the target accuracy is readily estimated by the initial run. The `VMC_workflow` then submits subsequent production VMC runs with the estimated necessary number of steps. Similar functionalities are also implemented in other workflow scripts such as `VMCopt_workflow`, `LRDMC_workflow`, and `LRDMCopt_workflow`. `TurboWorkflows` can solve the dependencies of a given set of workflows and manage sequential jobs. `Launcher` class accepts `workflows` as a list, solve the dependencies of the workflows, and submit independent sequential jobs simultaneously and independently. `Launcher` realises this feature by the so-called topological ordering of a Directed Acyclic Graph (DAG) and the build-in python module, `asyncio`. The following shows a workflow script to perform a sequential job, `PySCF` -> `TREXIO converion` -> `TurboRVB WF (JSD ansatz)` -> `VMC optimization (Jastrow factor optimization)` -> `VMC` -> `LRDMC` (`lattice space -> 0`). Finally, we get the extrapolated LRDMC energy of the water dimer.

`TurboWorkflows` manages file transfers as well as job submissions/collections from/to remote machines. `TurboWorkflows` supports job-queuing systems such as PBS and Slurm. `TurboWorkflows`relies on the `paramiko` module for its data transfer.

# Setup procedure of `TurboWorkflows`

When you run `TurboWorkflows` for the first time, `.turbofilemanager_config` directory is created at your home directory. You should edit `.turbofilemanager_config/machine_handler_env/machine_data.yaml`. One of the most important arguments is `file_manager_root`, which is explained later.

    # example of a remote computational server (e.g., a login node)
    henteko:
        machine_type: remote
        queuing : True
        computation: True
        ip: XXX.XX.XX.XX
        file_manager_root: /home/xxxx/xxxx/xxxx
        ssh_key: ~/.ssh/id_rsa
        ssh_option: -Y -A
        jobsubmit: /opt/pbs/bin/qsub
        jobcheck: /opt/pbs/bin/qstat
        jobdel: /opt/pbs/bin/qdel
        jobnum_index: 0

    # example of file-server
    nanashi:
        machine_type: remote
        queuing : False
        computation: False
        ip: XXX.XX.XX.XX
        file_manager_root: /mnt/aaaaa/bbbbb/ccccc
        ssh_key: ~/.ssh/id_rsa
        ssh_option: -Y -A

    # example of localhost (e.g., mac)
        localhost:
        machine_type: local
        queuing : False
        computation: True
        file_manager_root: /Users/xxxxx/yyyyy/zzzzz
        jobsubmit: bash
        jobcheck: ps
        jobnum_index: 1

If you install `TurboWorkflows` on a login node of a computation server (i.e., if you want to submit jobs via a job-queuing command directly from the login node where `TurboWorkflows` is installed), you can set up like

    # example of a login node
    localhost:
        machine_type: local
        queuing : True
        computation: True
        file_manager_root: /Users/xxxxxx/xxxxx/xxxxx
        jobsubmit: /opt/pbs/bin/qsub
        jobcheck: /opt/pbs/bin/qstat
        jobdel: /opt/pbs/bin/qdel
        jobnum_index: 0

`TurboWorkflows` works *only* in ``file_manager_root`` directory of the localhost.

You should also edit ``.turbofilemanager_config/{machine_name}/package.yaml``, ``turbofilemanager_config/{machine_name}/submit.sh``, ``turbofilemanager_config/{machine_name}/submit_nompi.sh``, and ``turbofilemanager_config/{machine_name}/queue_data.toml``.

    # package.yaml
    turborvb:
    name: turborvb
    binary_path:
        stable: /home/application/TurboRVB/bin
    binary_list:
        - turborvb-mpi.x
        - ...
    job_template
        mpi: submit.sh
        nompi: submit_nompi.sh

    # queue_data.toml
    [default] # queue_label
        # pre-defined variables
        mpi=false
        max_job_submit=1
        # other variables
        num_cores=1
        omp_num_threads=1
        nodes=1
        cpns=1
        mpi_per_node=1

    #submit_mpi.sh (PBS)
    #!/bin/bash
    #PBS -q _QUEUE_
    #PBS -N _JOBNAME_
    #PBS -l walltime=_MAX_TIME_
    #PBS -j oe
    #PBS -l select=_NODES_:ncpus=_CORES_PER_NODE_:mpiprocs=_MPI_PER_NODE_
    #PBS -V

    # Note:
    # The variables _xxx_ are replaced by job_manager.py. The pre-defined variables are
    # _INPUT_, _OUTPUT_, _PREOPTION_, _POSTOPTION_, _JOBNAME_, _BINARY_ROOT_, and _BINARY_.
    # Others defined in queue_data.toml (e.g. _NUM_CORE_) are also replaced by job_manager.py 
    # so that one can manually define other variables needed for submitting jobs to a queueing 
    # system.

    cd ${PBS_O_WORKDIR}

    export OMP_NUM_THREADS=_OMP_NUM_THREADS_

    CORES=_NUM_CORES_
    INPUT=_INPUT_
    OUTPUT=_OUTPUT_
    BINARY=_BINARY_ROOT_/_BINARY_

    mpirun -np $CORES $BINARY $PREOPTION < $INPUT $POSTOPTION > $OUTPUT

# Useful command-line tools of `TurboWorkflows`

`TurboWorkflows` provides two useful command-line tools:

- ``turbo-jobmanager`` (managing TurboWorkflows jobs)

## How to use ``turbo-jobmanager``

    # show running jobs in the current directory
    jobmanager show

    # show the detail of a job
    jobmanager show -id XX
    # here XX is obtained by the above show command.

    # delete running jobs
    jobmanager del -id XXXXX


# Quick use of `TurboWorkflows`

Installing from source

    git clone https://github.com/kousuke-nakano/turboworkflows
    cd turboworkflows
    pip install -e . or pip install .

# Examples
Examples are in the `examples` directory.

# Documentation for users
You can readily understand how to use `turboworkflows` by looking at the sample python scripts in the example directory. You can also see our tutorials [https://github.com/kousuke-nakano/turbotutorials].

# Documentation for developers
There is a Read the Docs in the `docs` directory, but still in progress. You can generate a html file using `sphinx`. Go to the `docs` directory, and type `make html`. The document is generated in `docs/_build/html`. `index.html` is the main page.

# How to contribute

Work on the development or on a new branch

    git merge <new branch> devel # if you work on a new branch.
    git push origin devel

Check the next-version version

    # Confirm the version number via `setuptools-scm`
    python -m setuptools_scm
    e.g., 1.1.4.dev28+gceef293.d20221123 -> <next-version> = v1.1.4 or v1.1.4-alpha(for pre-release)

Add and push with the new tag

    # Push with tag
    git tag <next-version>  # e.g., git tag v1.1.4  # Do not forget "v" before the version number!
    git push origin development --tags  # or to the new branch

Send a pull request to the main branch on GitHub.

# Reference
K. Nakano et al., [TurboGenius: Python suite for high-throughput calculations of ab initio quantum Monte Carlo methods](https://doi.org/10.1063/5.0179003), *J. Chem. Phys.* 159, 224801 (2023).

K. Nakano et al., TurboWorkflows: Benchmarking ab initio Quantum Monte Carlo Methods via high-throughput calculations, *in preparation* (2024).
