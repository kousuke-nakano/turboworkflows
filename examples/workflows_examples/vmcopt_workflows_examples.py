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
from turboworkflows.workflow_vmcopt import VMCopt_workflow
from turboworkflows.utils_turboworkflows.turboworkflows_env import turbo_workflows_root

prefix="vmcopt-workflows"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

vmcopt_workflow=VMCopt_workflow(
    ## job
    server_machine_name="localhost",
    cores=1,
    openmp=1,
    queue="DEFAULT",
    version="stable",
    sleep_time=30, # sec.
    jobpkl_name="job_manager",
    ## vmcopt
    vmcopt_max_continuation=2,
    vmcopt_pkl_name="vmcopt_genius",
    vmcopt_target_error_bar=1.0e-2,  # Ha
    vmcopt_trial_optsteps=10,
    vmcopt_trial_steps=10,
    vmcopt_production_optsteps=10,
    vmcopt_optwarmupsteps_ratio=0.8,
    vmcopt_bin_block=5,
    vmcopt_warmupblocks=0,
    vmcopt_optimizer="lr",
    vmcopt_learning_rate=0.35,
    vmcopt_regularization=0.001,
    vmcopt_onebody=True,
    vmcopt_twobody=True,
    vmcopt_det_mat=False,
    vmcopt_jas_mat=True,
    vmcopt_det_basis_exp=False,
    vmcopt_jas_basis_exp=False,
    vmcopt_det_basis_coeff=False,
    vmcopt_jas_basis_coeff=False,
    vmcopt_num_walkers = -1, # default -1 -> num of MPI process.
    vmcopt_twist_average=False,
    vmcopt_kpoints=[],
    vmcopt_maxtime=172000,
)

vmcopt_workflow.launch()