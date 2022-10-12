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
from turboworkflows.workflow_lrdmcopt import LRDMCopt_workflow
from turboworkflows.utils_turboworkflows.turboworkflows_env import turbo_workflows_root

# turbo-genius packages
from turbogenius.convertfort10mol_genius import Convertfort10mol_genius

prefix="lrdmcopt-workflows"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

lrdmcopt_workflow=LRDMCopt_workflow(
    ## job
    server_machine_name="localhost",
    cores=1,
    openmp=1,
    queue="DEFAULT",
    version="stable",
    sleep_time=30, # sec.
    jobpkl_name="job_manager",
    ## lrdmcopt
    lrdmcopt_max_continuation=2,
    lrdmcopt_pkl_name="lrdmcopt_genius",
    lrdmcopt_target_error_bar=1.0e-1,  # Ha
    lrdmcopt_trial_optsteps=2,
    lrdmcopt_trial_steps=100,
    lrdmcopt_production_optsteps=10,
    lrdmcopt_optwarmupsteps_ratio=0.8,
    lrdmcopt_bin_block=5,
    lrdmcopt_warmupblocks=0,
    lrdmcopt_optimizer="sr",
    lrdmcopt_learning_rate=0.002,
    lrdmcopt_regularization=0.001,
    lrdmcopt_alat=-0.30,
    lrdmcopt_trial_etry=-17.60,
    lrdmcopt_nonlocalmoves="dlatm",  # tmove, dla, dlatm
    lrdmcopt_onebody=False,
    lrdmcopt_twobody=False,
    lrdmcopt_det_mat=True,
    lrdmcopt_jas_mat=False,
    lrdmcopt_det_basis_exp=False,
    lrdmcopt_jas_basis_exp=False,
    lrdmcopt_det_basis_coeff=False,
    lrdmcopt_jas_basis_coeff=False,
    lrdmcopt_num_walkers = -1, # default -1 -> num of MPI process.
    lrdmcopt_twist_average=False,
    lrdmcopt_kpoints=[],
    lrdmcopt_maxtime=172000,
)

lrdmcopt_workflow.launch()
