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
from turboworkflows.workflow_vmc import VMC_workflow
from turboworkflows.utils_turboworkflows.turboworkflows_env import turbo_workflows_root

prefix="vmc-workflows"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

vmc_workflow=VMC_workflow(
    ## job
    server_machine_name="localhost",
    cores=1,
    openmp=1,
    queue="DEFAULT",
    version="stable",
    sleep_time=30, # sec.
    jobpkl_name="job_manager",
    ## vmc
    vmc_max_continuation=3,
    vmc_pkl_name="vmc_genius",
    vmc_target_error_bar=1.0e-2, # Ha
    vmc_trial_steps= 150,
    vmc_bin_block = 10,
    vmc_warmupblocks = 5,
    vmc_num_walkers = -1, # default -1 -> num of MPI process.
    vmc_twist_average=False,
    vmc_kpoints=[],
    vmc_force_calc_flag=False,
    vmc_maxtime=172000,
)

vmc_workflow.launch()