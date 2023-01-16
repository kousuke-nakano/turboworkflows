# TurboWorkflows

<img src="logo/turboworkflows_logo.jpg" width="70%">

![license](https://img.shields.io/github/license/kousuke-nakano/turboworkflows) ![release](https://img.shields.io/github/release/kousuke-nakano/turboworkflows/all.svg) ![fork](https://img.shields.io/github/forks/kousuke-nakano/turboworkflows?style=social) ![stars](https://img.shields.io/github/stars/kousuke-nakano/turboworkflows?style=social)

`TurboWorkflows` is a python package realizing high-throuput quantum Monte Carlo calculations with the SISSA ab-initio quantum Monte Carlo code, `TurboRVB`.

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

`Turboworkflows` provides a more sophisticated way to realize workflows by combining `TurboGenius` with a file/job managing package, `TurboFilemanager` [https://github.com/kousuke-nakano/turbofilemanager]. In `Turboworkflows`, each workflow class inherits the parent Workflow class with options useful for a QMC calculation. For instance, in the `VMC_workflow`, a user can specify a target accuracy (i.e., statistical error) of a VMC calculation. The `VMC_workflow` first submits an initial VMC run to a machine with the specified MPI and OpenMP processes to get a stochastic error bar per Monte Carlo step. Since the error bar is inversely proportional to the square root of the number of Monte Carlo samplings, the necessary steps to achieve the target accuracy is readily estimated by the initial run. The `VMC_workflow` then submits subsequent production VMC runs with the estimated necessary number of steps. Similar functionalities are also implemented in other workflow scripts such as `VMCopt_workflow`, `LRDMC_workflow`, and `LRDMCopt_workflow`. `TurboWorkflows` can solve the dependencies of a given set of workflows and manage sequential jobs. `Launcher` class accepts `workflows` as a list, solve the dependencies of the workflows, and submit independent sequential jobs simultaneously and independently. `Launcher` realises this feature by the so-called topological ordering of a Directed Acyclic Graph (DAG) and the build-in python module, `asyncio`. The following shows a workflow script to perform a sequential job, `PySCF` -> `TREXIO converion` -> `TurboRVB WF (JSD ansatz)` -> `VMC optimization (Jastrow factor optimization)` -> `VMC` -> `LRDMC` (`lattice space -> 0`). Finally, we get the extrapolated LRDMC energy of the water dimer.

# Quick use of `TurboWorkflows`

Installing from source

    git clone https://github.com/kousuke-nakano/turboworkflows
    cd turboworkflows
    pip install -e . or pip install .

# Examples
Examples are in the `examples` directory.

# Documentation for users
You can readily understand how to use `turboworkflows` by looking at the sample python scripts in the example directory. 
You can also see our tutorials [https://github.com/kousuke-nakano/turbotutorials].

# Documentation for developers
There is a Read the Docs in the `docs` directory, but still in progress.
You can generate a html file using `sphinx`. Go to the `docs` directory, 
and type `make html`. The document is generated in `docs/_build/html`.
`index.html` is the main page.

# How to contribute

Work on the development or on a new branch

    git merge <new branch> development # if you work on a new branch.
    git push origin development

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
K. Nakano et. al in prepareation (2023).
