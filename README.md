# TurboWorkflows

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
One can manage any job of `TurboRVB` on python scripts, or on your terminal using the provided command line tool `turbogenius`. 

# Quick use of `TurboWorkflows`

Installing from source

    git clone git@git-scm.sissa.it:sorella/turbo_workflows.git
    cd turboworkflows
    pip install -e . or pip install .

# Examples
Examples are in the `tests` directory.

# Documentation
There is a Read the Docs in the `docs` directory, but still in progress.
You can generate a html file using `sphinx`. Go to the `docs` directory, 
and type `make html`. The document is generated in `docs/_build/html`.
`index.html` is the main page.

# Reference
K. Nakano et. al in prepareation (2022).
