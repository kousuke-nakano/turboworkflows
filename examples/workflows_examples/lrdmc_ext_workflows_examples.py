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
from turboworkflows.workflow_lrdmc_ext import LRDMC_ext_workflow
from turboworkflows.utils_turboworkflows.turboworkflows_env import turbo_workflows_root

# turbo-genius packages
from turbogenius.convertfort10mol_genius import Convertfort10mol_genius

prefix="lrdmc-ext-workflows"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

lrdmc_ext_workflow=LRDMC_ext_workflow(
    ## job
    server_machine_name="localhost",
    cores=1,
    openmp=1,
    queue="DEFAULT",
    version="stable",
    sleep_time=30, # sec.
    jobpkl_name="job_manager",
    ## lrdmc
    lrdmc_max_continuation=3,
    lrdmc_pkl_name="lrdmc_genius",
    lrdmc_target_error_bar=1.0e-2, # Ha
    lrdmc_trial_steps= 150,
    lrdmc_bin_block = 10,
    lrdmc_warmupblocks = 5,
    lrdmc_correcting_factor=10,
    lrdmc_trial_etry=-17.208,
    lrdmc_alat_list=[-0.30, -0.40, -0.50],
    lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
    lrdmc_num_walkers = -1, # default -1 -> num of MPI process.
    lrdmc_twist_average=False,
    lrdmc_kpoints=[],
    lrdmc_force_calc_flag=False,
    lrdmc_maxtime=172000,
)

lrdmc_ext_workflow.launch()