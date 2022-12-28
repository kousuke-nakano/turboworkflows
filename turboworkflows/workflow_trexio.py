#!/usr/bin/env python
# coding: utf-8

# pyscf -> TREXIO

# python packages
import numpy as np
import os, sys
import shutil
import pickle
import numpy as np
import time
import asyncio
import glob
import pathlib

#Logger
from logging import config, getLogger, StreamHandler, Formatter, FileHandler
logger = getLogger('Turbo-Workflows').getChild(__name__)

# turboworkflow packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from workflow_encapsulated import Workflow
from pyscf_tools.pyscf_to_trexio import pyscf_to_trexio
from utils_turboworkflows.turboworkflows_env import turbo_workflows_root, turbo_workflows_source_root

# turbo-genius packages
from turbogenius.trexio_to_turborvb import trexio_to_turborvb_wf
from turbogenius.trexio_wrapper import Trexio_wrapper_r

# pyturbo package
from pyturbo.basis_set import Det_Basis_sets, Jas_Basis_sets
from pyturbo.io_fort10 import IO_fort10

# jobmanager
from turbofilemanager.job_manager import Job_submission

class TREXIO_convert_to_turboWF(Workflow):
    def __init__(self,
        trexio_filename="trexio.hdf5",
        twist_average=False,
        jastrow_basis_dict={},
        max_occ_conv=0,
        mo_num_conv=-1,
        only_mol=True,
        trexio_rerun=False,
        trexio_pkl_name="trexio_genius",
                 ):
        self.trexio_filename=trexio_filename
        self.twist_average=twist_average
        self.jastrow_basis_dict=jastrow_basis_dict
        self.max_occ_conv = max_occ_conv
        self.mo_num_conv = mo_num_conv
        self.only_mol=only_mol
        self.trexio_rerun=trexio_rerun
        self.trexio_pkl_name=trexio_pkl_name
        ## return values
        self.status = "init"
        self.output_files = []
        self.output_values = {}

    async def async_launch(self):
        ###############################################
        # Start a workflow
        ###############################################
        self.root_dir=os.getcwd()
        logger.info(f"Current dir = {self.root_dir}")
        self.trexio_pkl = f"{self.trexio_pkl_name}.pkl"

        #******************
        #! trexio
        #******************
        os.chdir(self.root_dir)
        self.trexio_dir=os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.trexio_dir, "pkl")
        logger.info(f"Project root dir = {self.trexio_dir}")
        logger.info(f"Start: TREXIO calculation")

        if self.trexio_rerun or not os.path.isfile(os.path.join(self.pkl_dir, self.trexio_pkl)):
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.trexio_dir)

            if self.twist_average:
                with open(os.path.join(self.trexio_dir, "kp_info.dat"), "r") as f:
                    lines = f.readlines()
                k_num = len(lines) - 1
            else:
                k_num = 1

            for num in range(k_num):
                if self.twist_average:
                    filename = f"k{num}_" + os.path.basename(self.trexio_filename)
                else:
                    filename = os.path.basename(self.trexio_filename)
                if len(self.jastrow_basis_dict) != 0:
                    trexio_r = Trexio_wrapper_r(trexio_file=os.path.join(self.trexio_dir, filename))
                    jastrow_basis_list = [self.jastrow_basis_dict[element] for element in trexio_r.labels_r]
                    jas_basis_sets = Jas_Basis_sets.parse_basis_sets_from_texts(jastrow_basis_list, format="gamess")
                else:
                    jas_basis_sets = Jas_Basis_sets()

                ## trexio -> turborvb_wf
                trexio_to_turborvb_wf(trexio_file=os.path.join(self.trexio_dir, filename), jas_basis_sets=jas_basis_sets,
                                      max_occ_conv=self.max_occ_conv, mo_num_conv=self.mo_num_conv, only_mol=self.only_mol)

                if self.twist_average:
                    turborvb_scratch_dir = os.path.join(self.trexio_dir, "turborvb.scratch")
                    os.makedirs(turborvb_scratch_dir, exist_ok=True)
                    shutil.move(os.path.join(self.trexio_dir, "fort.10"), os.path.join(turborvb_scratch_dir, "fort.10_{:0>6}".format(num)))

            if self.twist_average:
                shutil.copy(os.path.join(turborvb_scratch_dir, "fort.10_{:0>6}".format(0)), os.path.join(self.trexio_dir, "fort.10"))

            if self.twist_average:
                with open(os.path.join(self.trexio_dir, "kp_info.dat"), "r") as f:
                    lines = f.readlines()
                    kpoints_up = []
                    kpoints_dn = []
                    for line in lines[1:]:
                        k_index, kx, ky, kz = line.split()
                        wk = 1.0
                        kpoints_up.append([float(kx), float(ky), float(kz), float(wk)])
                        kpoints_dn.append([float(kx), float(ky), float(kz), float(wk)])
                    self.kpoints = [kpoints_up, kpoints_dn]
                    self.output_values["kpoints"] = self.kpoints

            #mo occ
            if self.twist_average:
                pass # to be implemented!!
            else:
                filename = os.path.basename(self.trexio_filename)
                trexio_r = Trexio_wrapper_r(trexio_file=os.path.join(self.trexio_dir, filename))
                self.output_values["mo_occ"]=trexio_r.mo_occupation

            with open(os.path.join(self.trexio_dir, self.trexio_pkl), "wb") as f:
                pickle.dump("dummy", f)

        else:
            logger.info(f"{self.trexio_pkl} exists.")
            logger.info(f"Skip: TREXIO calculation")
            if self.twist_average:
                with open(os.path.join(self.trexio_dir, "kp_info.dat"), "r") as f:
                    lines=f.readlines()
                    kpoints_up = []
                    kpoints_dn = []
                    for line in lines[1:]:
                        k_index, kx, ky, kz = line.split()
                        wk = 1.0
                        kpoints_up.append([float(kx), float(ky), float(kz), float(wk)])
                        kpoints_dn.append([float(kx), float(ky), float(kz), float(wk)])
                    self.kpoints = [kpoints_up, kpoints_dn]
                    self.output_values["kpoints"] = self.kpoints
            #mo occ
            if self.twist_average:
                pass # to be implemented!!
            else:
                filename = os.path.basename(self.trexio_filename)
                trexio_r = Trexio_wrapper_r(trexio_file=os.path.join(self.trexio_dir, filename))
                self.output_values["mo_occ"]=trexio_r.mo_occupation

        with open(os.path.join(self.trexio_dir, self.trexio_pkl), "wb") as f:
            pickle.dump("dummy", f)
        with open(os.path.join(self.pkl_dir, self.trexio_pkl), "wb") as f:
            pickle.dump("dummy", f)

        logger.info(f"End: TREXIO conversion"); os.chdir(self.trexio_dir)
        await asyncio.sleep(1)

        os.chdir(self.root_dir)
        self.status="success"
        p_list=[pathlib.Path(ob) for ob in glob.glob(os.path.join(self.root_dir,'*'))]
        self.output_files = [str(p.resolve().relative_to(self.root_dir)) for p in p_list]
        return self.status, self.output_files, self.output_values

if __name__ == "__main__":
    logger = getLogger("Turbo-Workflows")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

