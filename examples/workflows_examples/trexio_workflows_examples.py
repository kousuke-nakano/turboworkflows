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
from turboworkflows.workflow_trexio import TREXIO_convert_to_turboWF
from turboworkflows.utils_turboworkflows.turboworkflows_env import turbo_workflows_root

prefix="trexio-workflows"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

trexio_workflow=TREXIO_convert_to_turboWF(
    trexio_filename="trexio.hdf5",
    twist_average=False,
    jastrow_basis_dict={},
    max_occ_conv=0,
    mo_num_conv=-1,
    trexio_rerun=False,
    trexio_pkl_name="trexio_genius"
)

trexio_workflow.launch()