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
from turboworkflows.workflow_collections import Jastrowcopy_workflow
from turboworkflows.workflow_collections import init_occ_workflow
from turboworkflows.workflow_collections import Makefort10_workflow
from turbogenius.trexio_wrapper import Trexio_wrapper_r
from turboworkflows.utils_turboworkflows.turboworkflows_env import turbo_workflows_root

prefix="jascopy_workflow"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

jascopy_workflow=Jastrowcopy_workflow(
    ## copyjastrow
    jastrowcopy_rerun=False,
    jastrowcopy_pkl_name="jastrowcopy_genius",
    jastrowcopy_fort10_to="fort.10",
    jastrowcopy_fort10_from="fort.10_new",
    jastrowcopy_twist_average=False
)
jascopy_workflow.launch()



prefix="jascopy_twist_workflow"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

jascopy_workflow=Jastrowcopy_workflow(
    ## copyjastrow
    jastrowcopy_rerun=False,
    jastrowcopy_pkl_name="jastrowcopy_genius",
    jastrowcopy_fort10_to="fort.10",
    jastrowcopy_fort10_from="fort.10_new",
    jastrowcopy_twist_average=True
)
jascopy_workflow.launch()



prefix="init_occ_workflow"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

trexio_r = Trexio_wrapper_r(trexio_file="trexio.hdf5")
mo_occ = trexio_r.mo_occupation

init_occ_workflow=init_occ_workflow(
                init_occ_rerun=True,
                init_occ_pkl_name="init_occ_genius",
                mo_occ_fixed_list=[0],
                mo_occ_thr=1.0e-3,
                mo_num_conv=-1,
                mo_occ=mo_occ,
                mo_occ_delta=0.05
)
init_occ_workflow.launch()



prefix="makefort10-workflows"
example_root_dir=os.path.join(turbo_workflows_root, "examples", "workflows_examples")
if os.path.isdir(os.path.join(example_root_dir, prefix)):
    shutil.rmtree(os.path.join(example_root_dir, prefix))
shutil.copytree(os.path.join(example_root_dir, "all_input_files", prefix), os.path.join(example_root_dir, prefix))
os.chdir(os.path.join(example_root_dir, prefix))

jastrow_basis_dict = {
    'C':
        """
        S  1
        1       1.637494  1.00000000
        S  1
        1       0.921552  1.00000000
        S  1
        1       0.409924  1.00000000
        P  1
        1       0.935757  1.00000000
        """,
}

makefort10_workflow=Makefort10_workflow(
    structure_file="Diamond.cif",
    ## job
    makefort10_rerun=True,
    makefort10_pkl_name="makefort10",
    ## genius-related arguments
    supercell=[1, 1, 1],
    det_basis_set="cc-pVQZ",
    jas_basis_set=[jastrow_basis_dict["C"]]*8,
    det_contracted_flag=True,
    jas_contracted_flag=True,
    all_electron_jas_basis_set=True,
    pseudo_potential="ccECP",
    det_cut_basis_option=True,
    jas_cut_basis_option=False,
    jastrow_type=-6,
    complex=False,
    phase_up=[0.0, 0.0, 0.0],
    phase_dn=[0.0, 0.0, 0.0],
    neldiff=0
)

makefort10_workflow.launch()