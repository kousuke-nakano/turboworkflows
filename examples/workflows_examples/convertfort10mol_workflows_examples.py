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
from turboworkflows.workflow_convertfort10mol import Convertfort10mol_workflow
from turboworkflows.utils_turboworkflows.turboworkflows_env import turbo_workflows_root

# turbo-genius packages
from turbogenius.convertfort10mol_genius import Convertfort10mol_genius

prefix="convertfort10mol-workflows"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

convertfort10mol_workflow=Convertfort10mol_workflow(
    ## job
    server_machine_name="localhost",
    cores=1,
    openmp=1,
    queue="SINGLE",
    version="stable",
    sleep_time=180,  # sec.
    jobpkl_name="job_manager",
    # convertfort10mol
    convertfort10mol_rerun=False,
    convertfort10mol_pkl_name="convertfort10mol_genius",
    add_random_mo=True,
    grid_size=0.10,
    additional_mo=0
)

convertfort10mol_workflow.launch()