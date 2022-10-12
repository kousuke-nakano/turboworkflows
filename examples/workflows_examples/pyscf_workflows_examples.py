#!python
# -*- coding: utf-8 -*-

#python modules
import os, sys, shutil

#Logger
from logging import config, getLogger, StreamHandler, Formatter, FileHandler
logger = getLogger("Turbo-Workflows")
logger.setLevel("INFO")
stream_handler = StreamHandler()
stream_handler.setLevel("DEBUG")
handler_format = Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# turboworkflows packages
from turboworkflows.workflow_pyscf import PySCF_workflow
from turboworkflows.utils_turboworkflows.turboworkflows_env import turbo_workflows_root

prefix="pyscf-workflows"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

pyscf_workflow=PySCF_workflow(
    ## structure file (mandatory)
    structure_file="water.xyz",
    ## job
    server_machine_name="localhost",
    cores=1,
    openmp=1,
    queue="DEFAULT",
    version="stable",
    sleep_time=10,  # sec.
    jobpkl_name="job_manager",
    ## pyscf
    pyscf_rerun=False,
    pyscf_pkl_name="pyscf_genius",
    charge=0,
    spin=0,
    basis="ccecp-ccpvtz",  # defined below
    ecp="ccecp",  # defined below
    scf_method="DFT",  # HF or DFT
    dft_xc="LDA_X,LDA_C_PZ",
    mp2_flag=False,
    pyscf_output="out.pyscf",
    pyscf_chkfile="pyscf.chk",
    solver_newton=False,
    twist_average=False,
    exp_to_discard=0.10,
    kpt=[0.0, 0.0, 0.0],  # scaled_kpts!! i.e., crystal coord.
    kpt_grid=[1, 1, 1]
)

pyscf_workflow.launch()

