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

`TurboWorkflows` manages file transfers as well as job submissions/collections from/to remote machines. `TurboWorkflows` supports job-queuing systems such as PBS and Slurm. `TurboWorkflows`relies on `rsync` and `scp` commands so that it works on a Linux-based machine.

# Setup procedure of `TurboWorkflows`

When you run `TurboWorkflows` for the first time, `turbofilemanager_config` directory is created at your home directory. You should edit `turbofilemanager_config/machine_handler_env/machine_data.yaml`. Here, the most important argument is `file_manager_root`, which is explained later.

    # example of a remote computational server (e.g., a login node)
    henteko:
    machine_type: remote
    queuing : True
    computation: True
    ip: XXX.XX.XX.XX
    ssh_port: 22
    username: xxxx/xxxxx
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
    ssh_port: 22
    username: xxxxxx
    file_manager_root: /mnt/aaaaa/bbbbb/ccccc
    ssh_key: ~/.ssh/id_rsa
    ssh_option: -Y -A

    # example of localhost (e.g., mac)
    localhost:
    machine_type: local
    queuing : False
    computation: True
    username: None
    file_manager_root: /Users/xxxxx/yyyyy/zzzzz
    jobsubmit: bash
    jobcheck: ps
    jobnum_index: 1

If you install `TurboWorkflows` on a login node of a computation server (i.e., if you want to submit jobs via a job-queuing command directly from the node where `TurboWorkflows` is installed), you can set up like

    # example of a login node
    localhost:
    machine_type: local
    queuing : True
    computation: True
    username: None
    file_manager_root: /Users/xxxxxx/xxxxx/xxxxx
    jobsubmit: /opt/pbs/bin/qsub
    jobcheck: /opt/pbs/bin/qstat
    jobdel: /opt/pbs/bin/qdel
    jobnum_index: 0

Both `TurboWorkflows` works *only* in ``file_manager_root`` directory of the localhost.

You should also edit ``turbofilemanager_config/job_manager_env/machine_name/package.yaml``, ``turbofilemanager_config/job_manager_env/machine_name/submit.sh``, and ``turbofilemanager_config/job_manager_env/machine_name/queue_data.txt``.

    #package.yaml
    turborvb:
    name: turborvb
    binary_path:
        stable: /home/application/TurboRVB/bin
    binary_list:
        - turborvb-mpi.x
        - A
        - B
        - ...

    #queue_data.txt
    QUEUE   CORES  OMP  NODES  CPNS  MPI_PER_NODE  MAX_JOB_RUN  MAX_JOB_SUBMIT  MAX_TIME
    TINY     64    1    1       64    64           10           16               000:30:00
    SINGLE  128    1    1      128   128           10           16               168:00:00
    LONG    128    1    1      128   128           10           16               168:00:00
    SMALL   480    1    4      128   120           10           16               168:00:00
    SMALL   512    1    4      128   128           10           16               168:00:00

    #submit_mpi.sh (PBS)
    #!/bin/bash
    #PBS -q _QUEUE_
    #PBS -N _JOBNAME_
    #PBS -l walltime=_MAX_TIME_
    #PBS -j oe
    #PBS -l select=_NODES_:ncpus=_CORES_PER_NODE_:mpiprocs=_MPI_PER_NODE_
    #PBS -V

    # Note:
    # The variables _XXXXXX_ are replaced by job_manager.py.
    # Implemented arguments are:
    # QUEUE, JOBNAME, MAX_TIME, NODES, CORES_PER_NODE, MPI_PER_NODE
    # OMP_NUM_THREADS, NUM_CORES, INPUT, OUTPUT, BINARY_ROOT, BINARY
    # PREOPTION, POSTOPTION

    cd ${PBS_O_WORKDIR}

    export OMP_NUM_THREADS=_OMP_NUM_THREADS_

    CORES=_NUM_CORES_
    INPUT=_INPUT_
    OUTPUT=_OUTPUT_
    BINARY=_BINARY_ROOT_/_BINARY_

    mpirun -np $CORES $BINARY $PREOPTION < $INPUT $POSTOPTION > $OUTPUT

# Useful command-line tools of `TurboWorkflows`

`TurboWorkflows` provides two useful command-line tools:

- ``turbo-filemanager`` (managing file transfers)
- ``turbo-jobmanager`` (managing job submissions and collections)

## How to use the command-line tools

    turbo-filemanager put -s remoteserver # transfer files in the current dir. to a remote server
    turbo-filemanager get -s remoteserver # transfer files in the current dir. from a remote server

## How to use ``turbo-filemanager``

``turbo-filemanager`` implements ``put`` and ``get`` commands. The commands transfer files from/to the ``localhost`` to/from a specified ``remotehost``. Concerning the destination, ``file_manager_root`` of the ``localhost`` is replaced with that of the ``remotehost``. For instance, suppose you are in ``/Users/xxxxx/yyyyy/zzzzz/kk/ll`` on your ``localhost`` whose ``file_manager_root`` is ``/Users/xxxxx/yyyyy/zzzzz/``. When you transfer the files in the current directory on ``localhost`` to ``nanashi`` whose ``file_manager_root`` is ``/mnt/aaaaa/bbbbb/ccccc`` by the ``put`` command, all the files in ``/Users/xxxxx/yyyyy/zzzzz/kk/ll`` on ``localhost`` will be transfered to ``/mnt/aaaaa/bbbbb/ccccc/kk/ll`` on ``nanashi``.

A remotehost can be specified by ``-s`` option. You can see ``--help``.

``turbo-filemanager`` transfers all the files in the current directory. If you want to include/exclude specific files, you can use ``--include``/``--exclude`` options. You can see ``--help``.

## How to use ``turbo-jobmanager``

    # for submissions
    turbo-jobmanager toss -s remoteserver -p turborvb -core 144
    # the default binary is the first one on package.yaml
    # queue, omp, etc... is automatically chosen from queue_data.txt.
    # the default inputfile/outputfile name is input.in/out.o respectively.
    # you can see --help

    # you can explicitly specify them, e.g.,
    turbo-jobmanager toss -s remoteserver -p turborvb -core 144 -b prep-mpi.x -omp 2 -i prep.input -o out_prep -q SINGLE

    # for collections
    jobmanager fetch

    # check running jobs
    jobmanager stat -s remoteserver

    # delete running jobs
    jobmanager del -s remoteserver -id XXXXX

    # show running jobs in the current directory
    jobmanager show

    # show the detail of a job
    jobmanager show -id XX
    # here XX is obtained by the above show command.

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
