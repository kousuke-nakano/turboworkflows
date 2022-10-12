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
from turboworkflows.workflow_prep import DFT_workflow
from turboworkflows.utils_turboworkflows.turboworkflows_env import turbo_workflows_root

prefix="prep-workflows"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

prep_workflow=DFT_workflow(
    ## job
    server_machine_name="localhost",
    cores=1,
    openmp=1,
    queue="DEFAULT",
    version="stable",
    sleep_time=30, # sec.
    jobpkl_name="job_manager",
    ## prep
    dft_rerun = False,
    dft_pkl_name = "prep",
    dft_grid_size=[0.4, 0.4, 0.4],
    dft_lbox=[15.0, 15.0, 15.0],
    dft_smearing=0.0,
    dft_maxtime=172800,
    dft_maxit=2,
    dft_h_field=0.0,
    dft_magnetic_moment_list=[],
    dft_xc='lda',  # lda or lsda
    dft_twist_average=False,
    dft_kpoints=[]
)

prep_workflow.launch()
