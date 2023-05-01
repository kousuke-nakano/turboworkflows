#!/usr/bin/env python
# coding: utf-8

# python packages
import numpy as np
import os
import pickle
import asyncio
import glob
import pathlib
from typing import Optional

# Logger
from logging import getLogger, StreamHandler, Formatter

# turbo-genius packages
from turbogenius.makefort10_genius import Makefort10_genius
from turbogenius.convertfort10mol_genius import Convertfort10mol_genius
from turbogenius.tools_genius import copy_jastrow
from turbogenius.pyturbo.io_fort10 import IO_fort10
from turbogenius.wavefunction import Wavefunction

# turboworkflows packages
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from turboworkflows.workflow_encapsulated import Workflow


logger = getLogger("Turbo-Workflows").getChild(__name__)


# Jastrow copy
class Jastrowcopy_workflow(Workflow):
    def __init__(
        self,
        # copyjastrow
        jastrowcopy_rerun: bool = False,
        jastrowcopy_pkl_name: str = "jastrowcopy_genius",
        jastrowcopy_fort10_to: str = "fort.10",
        jastrowcopy_fort10_from: str = "fort.10_new",
        jastrowcopy_twist_average: bool = False,
    ):
        # copyjas
        self.jastrowcopy_rerun = jastrowcopy_rerun
        self.jastrowcopy_pkl_name = jastrowcopy_pkl_name
        self.jastrowcopy_fort10_to = jastrowcopy_fort10_to
        self.jastrowcopy_fort10_from = jastrowcopy_fort10_from
        self.jastrowcopy_twist_average = jastrowcopy_twist_average

        # return values
        self.status = "init"
        self.output_files = []
        self.output_values = {}

    async def async_launch(self):
        ###############################################
        # Start a workflow
        ###############################################
        self.root_dir = os.getcwd()
        logger.info(f"Current dir = {self.root_dir}")

        # ******************
        # copy Jastrow
        # ******************
        os.chdir(self.root_dir)
        self.jastrowcopy_dir = os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.jastrowcopy_dir, "pkl")
        logger.info(f"Project root dir = {self.jastrowcopy_dir}")
        self.jastrowcopy_pkl = f"{self.jastrowcopy_pkl_name}.pkl"

        if self.jastrowcopy_rerun or not os.path.isfile(
            os.path.join(self.pkl_dir, self.jastrowcopy_pkl)
        ):
            logger.info("Start: Jastrow copy")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.jastrowcopy_dir)

            copy_jastrow(
                fort10_to=self.jastrowcopy_fort10_to,
                fort10_from=self.jastrowcopy_fort10_from,
                twist_flag=self.jastrowcopy_twist_average,
            )

            with open(
                os.path.join(self.jastrowcopy_dir, self.jastrowcopy_pkl), "wb"
            ) as f:
                pickle.dump("dummy", f)
            with open(os.path.join(self.pkl_dir, self.jastrowcopy_pkl), "wb") as f:
                pickle.dump("dummy", f)

            os.chdir(self.root_dir)

        else:
            logger.info("Skip: Jastrow copy")

        logger.info("End: Jastrow copy workflow ends.")
        await asyncio.sleep(1)
        os.chdir(self.root_dir)

        self.status = "success"
        p_list = [
            pathlib.Path(ob) for ob in glob.glob(os.path.join(self.root_dir, "*"))
        ]
        self.output_files = [
            str(p.resolve().relative_to(self.root_dir)) for p in p_list
        ]
        return self.status, self.output_files, self.output_values


# init occ_workflow
class Init_occ_workflow(Workflow):
    def __init__(
        self,
        init_occ_rerun: bool = False,
        init_occ_pkl_name: str = "init_occ_genius",
        mo_occ_fixed_list: Optional[list] = None,
        mo_occ_fixed_occupied: bool = False,
        mo_occ_thr: float = 1.0e-3,
        mo_num_conv: int = -1,
        mo_occ: Optional[list] = None,
        mo_occ_delta: float = 0.05,
    ):
        if mo_occ_fixed_list is None:
            mo_occ_fixed_list = []
        if mo_occ is None:
            mo_occ = []
        self.init_occ_rerun = init_occ_rerun
        self.init_occ_pkl_name = init_occ_pkl_name
        self.mo_occ_fixed_list = mo_occ_fixed_list
        self.mo_occ_fixed_occupied = mo_occ_fixed_occupied
        self.mo_occ_thr = mo_occ_thr
        self.mo_num_conv = mo_num_conv
        self.mo_occ = mo_occ
        self.mo_occ_delta = mo_occ_delta

        # return values
        self.status = "init"
        self.output_files = []
        self.output_values = {}

    async def async_launch(self):
        ###############################################
        # Start a workflow
        ###############################################
        self.root_dir = os.getcwd()
        logger.info(f"Current dir = {self.root_dir}")

        # ******************
        # init_occ
        # ******************
        os.chdir(self.root_dir)
        self.init_occ_dir = os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.init_occ_dir, "pkl")
        logger.info(f"Project root dir = {self.init_occ_dir}")
        self.init_occ_pkl = f"{self.init_occ_pkl_name}.pkl"

        if self.init_occ_rerun or not os.path.isfile(
            os.path.join(self.pkl_dir, self.init_occ_pkl)
        ):
            logger.info("Start: initialization occ.")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.init_occ_dir)

            # important manipulation of fort.10!!
            io_fort10 = IO_fort10("fort.10")
            # here, specify how many MOs are used!!
            if self.mo_occ_thr == 0 and self.mo_num_conv == -1:  # default
                logger.info(
                    f"All eigenvalues of the MOs {len(self.mo_occ)} are used for the LRDMC opt."
                )
                mo_index = list(range(len(self.mo_occ)))
            elif self.mo_occ_thr != 0 and self.mo_num_conv != -1:
                logger.error(
                    "mo_occ_thr and mo_num_conv options cannot be used at the same time."
                )
                raise ValueError
            else:
                if self.mo_occ_thr != 0:
                    mo_index = []
                    logger.info(
                        f"eigenvalues of the MOs < mo_occ_thr:{self.mo_occ_thr} will be used."
                    )
                    for i in range(len(self.mo_occ)):
                        mo_index.append(i)
                        if (
                            self.mo_occ[i] >= self.mo_occ_thr
                            and self.mo_occ[i + 1] < self.mo_occ_thr
                        ):
                            logger.info(f"mo_occ < {self.mo_occ_thr} is 1-{i + 1}")
                            break

                else:  # mo_num_conv != -1:
                    logger.info(
                        f"eigenvalues of the MOs mo_num_conv = {self.mo_num_conv} will be used."
                    )
                    mo_index = list(range(self.mo_num_conv))

            # fixed all occupied orbitals
            if self.mo_occ_fixed_occupied:
                for i_nel in range(io_fort10.f10header.neldn):
                    if i_nel not in self.mo_occ_fixed_list:
                        self.mo_occ_fixed_list.append(i_nel)

                for nel_diff in range(
                    io_fort10.f10header.nelup - io_fort10.f10header.neldn
                ):
                    i_nel = len(io_fort10.f10detmat_sym.constraint_num) - nel_diff - 1
                    if i_nel not in self.mo_occ_fixed_list:
                        self.mo_occ_fixed_list.append(i_nel)

            # here, the ascending order is assumed.
            # if fort.10 generated by makefort10.x with
            # only_molecular=.true, but not general.
            # for the time being, it seems ok...
            sym_const_num_list = io_fort10.f10detmat_sym.constraint_num
            coeff_real = io_fort10.f10detmatrix.coeff_real
            for i in range(len(mo_index)):
                if i in self.mo_occ_fixed_list:
                    coeff_real[i] = 1.0
                    sym_const_num_list[i] = -1 * np.abs(sym_const_num_list[i])
                    continue
                sym_const_num_list[i] = np.abs(sym_const_num_list[i])
                if coeff_real[i] > 0.5:
                    coeff_real[i] = 1.0 - self.mo_occ_delta
                else:
                    coeff_real[i] = 0.0 + self.mo_occ_delta

            """ commented out for the time being
            for i in range(len(mo_index), len(self.mo_occ)):
                logger.info(f"i={i}")
                sym_const_num_list[i] = -1 * np.abs(sym_const_num_list[i])
                coeff_real[i] = 0.0
            """

            io_fort10.f10detmat_sym.constraint_num = sym_const_num_list
            io_fort10.f10detmatrix.coeff_real = coeff_real
            logger.info(f"Replaced coeff_real={io_fort10.f10detmatrix.coeff_real}")

            with open(os.path.join(self.init_occ_dir, self.init_occ_pkl), "wb") as f:
                pickle.dump("dummy", f)
            with open(os.path.join(self.pkl_dir, self.init_occ_pkl), "wb") as f:
                pickle.dump("dummy", f)

            os.chdir(self.root_dir)

        else:
            logger.info("Skip: initialization occ.")

        logger.info("End: initialization occ. workflow ends.")
        await asyncio.sleep(1)
        os.chdir(self.root_dir)

        self.status = "success"
        p_list = [
            pathlib.Path(ob) for ob in glob.glob(os.path.join(self.root_dir, "*"))
        ]
        self.output_files = [
            str(p.resolve().relative_to(self.root_dir)) for p in p_list
        ]
        return self.status, self.output_files, self.output_values


# makefort10 workflow
class Makefort10_workflow(Workflow):
    def __init__(
        self,
        structure_file: str,
        # job
        makefort10_rerun: bool = False,
        makefort10_pkl_name: str = "makefort10",
        # genius-related arguments
        supercell: Optional[list] = None,
        det_basis_set: str = "cc-pVQZ",
        jas_basis_set: str = "cc-pVQZ",
        det_contracted_flag: bool = True,
        jas_contracted_flag: bool = True,
        all_electron_jas_basis_set: bool = True,
        pseudo_potential: bool = None,
        det_cut_basis_option: bool = False,
        det_exp_to_discard: float = 0.00,
        jas_cut_basis_option: bool = False,
        jastrow_type: int = -6,
        complex: bool = False,
        phase_up: Optional[list] = None,
        phase_dn: Optional[list] = None,
        same_phase_up_dn: bool = False,
        neldiff: int = 0,
    ):
        if supercell is None:
            supercell = [1, 1, 1]
        if phase_up is None:
            phase_up = [0.0, 0.0, 0.0]
        if phase_dn is None:
            phase_dn = [0.0, 0.0, 0.0]
        # job
        self.makefort10_rerun = makefort10_rerun
        self.makefort10_pkl_name = makefort10_pkl_name
        # genius-related arguments
        self.structure_file = structure_file
        self.supercell = supercell
        self.det_basis_set = det_basis_set
        self.jas_basis_set = jas_basis_set
        self.det_contracted_flag = det_contracted_flag
        self.jas_contracted_flag = jas_contracted_flag
        self.all_electron_jas_basis_set = all_electron_jas_basis_set
        self.pseudo_potential = pseudo_potential
        self.det_cut_basis_option = det_cut_basis_option
        self.det_exp_to_discard = det_exp_to_discard
        self.jas_cut_basis_option = jas_cut_basis_option
        self.jastrow_type = jastrow_type
        self.complex = complex
        self.phase_up = phase_up
        self.phase_dn = phase_dn
        self.same_phase_up_dn = same_phase_up_dn
        self.neldiff = neldiff
        # return values
        self.status = "init"
        self.output_files = []
        self.output_values = {}

    async def async_launch(self):
        ###############################################
        # Start a workflow
        ###############################################
        self.root_dir = os.getcwd()
        logger.info(f"Current dir = {self.root_dir}")

        # ******************
        # genius-related arguments
        # ******************
        os.chdir(self.root_dir)
        self.makefort10_dir = os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.makefort10_dir, "pkl")
        self.makefort10_pkl = f"{self.makefort10_pkl_name}.pkl"
        logger.info(f"Project root dir = {self.makefort10_dir}")

        #####
        # main part
        #####
        if self.makefort10_rerun or not os.path.isfile(
            os.path.join(self.pkl_dir, self.makefort10_pkl)
        ):
            logger.info("Start: makefort10")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.makefort10_dir)

            makefort10 = Makefort10_genius(
                structure_file=self.structure_file,
                supercell=self.supercell,
                det_basis_set=self.det_basis_set,
                jas_basis_set=self.jas_basis_set,
                det_contracted_flag=self.det_contracted_flag,
                jas_contracted_flag=self.jas_contracted_flag,
                all_electron_jas_basis_set=self.all_electron_jas_basis_set,
                pseudo_potential=self.pseudo_potential,
                det_cut_basis_option=self.det_cut_basis_option,
                det_exp_to_discard=self.det_exp_to_discard,
                jas_cut_basis_option=self.jas_cut_basis_option,
                jastrow_type=self.jastrow_type,
                complex=self.complex,
                phase_up=self.phase_up,
                phase_dn=self.phase_dn,
                same_phase_up_dn=self.same_phase_up_dn,
                neldiff=self.neldiff,
            )

            makefort10.run_all()

            with open(
                os.path.join(self.makefort10_dir, self.makefort10_pkl), "wb"
            ) as f:
                pickle.dump("dummy", f)
            with open(os.path.join(self.pkl_dir, self.makefort10_pkl), "wb") as f:
                pickle.dump("dummy", f)

            os.chdir(self.root_dir)

        else:
            logger.info("Skip: makefort10")

        logger.info("End: makefort10 workflow ends.")
        await asyncio.sleep(1)
        os.chdir(self.root_dir)

        #####
        #
        #####

        self.status = "success"
        p_list = [
            pathlib.Path(ob) for ob in glob.glob(os.path.join(self.root_dir, "*"))
        ]
        self.output_files = [
            str(p.resolve().relative_to(self.root_dir)) for p in p_list
        ]
        return self.status, self.output_files, self.output_values


# convertfort10mol
class Convertfort10mol_workflow(Workflow):
    def __init__(
        self,
        convertfort10mol_rerun: bool = False,
        convertfort10mol_pkl_name: str = "convertfort10mol_genius",
        add_random_mo: bool = True,
        grid_size: float = 0.10,
        additional_mo: float = 0,
    ):

        # convertfort10mol
        self.convertfort10mol_rerun = convertfort10mol_rerun
        self.convertfort10mol_pkl_name = convertfort10mol_pkl_name
        self.add_random_mo = add_random_mo
        self.grid_size = grid_size
        self.additional_mo = additional_mo

        # return values
        self.status = "init"
        self.output_files = []
        self.output_values = {}

    async def async_launch(self):
        ###############################################
        # Start a workflow
        ###############################################
        self.root_dir = os.getcwd()
        logger.info(f"Current dir = {self.root_dir}")

        # ******************
        # convertfort10mol
        # ******************
        os.chdir(self.root_dir)
        self.convertfort10mol_dir = os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.convertfort10mol_dir, "pkl")
        logger.info(f"Project root dir = {self.convertfort10mol_dir}")
        self.convertfort10mol_pkl = f"{self.convertfort10mol_pkl_name}.pkl"

        if self.convertfort10mol_rerun or not os.path.isfile(
            os.path.join(self.pkl_dir, self.convertfort10mol_pkl)
        ):
            logger.info("Start: convertfort10mol")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.convertfort10mol_dir)

            convertfort10mol = Convertfort10mol_genius(
                add_random_mo=self.add_random_mo,
                grid_size=self.grid_size,
                additional_mo=self.additional_mo,
            )

            convertfort10mol.run_all()

            with open(
                os.path.join(self.convertfort10mol_dir, self.convertfort10mol_pkl),
                "wb",
            ) as f:
                pickle.dump("dummy", f)
            with open(os.path.join(self.pkl_dir, self.convertfort10mol_pkl), "wb") as f:
                pickle.dump("dummy", f)

            os.chdir(self.root_dir)

        else:
            logger.info("Skip: convertfort10mol")

        logger.info("End: convertfort10mol workflow ends.")
        await asyncio.sleep(1)
        os.chdir(self.root_dir)

        self.status = "success"
        p_list = [
            pathlib.Path(ob) for ob in glob.glob(os.path.join(self.root_dir, "*"))
        ]
        self.output_files = [
            str(p.resolve().relative_to(self.root_dir)) for p in p_list
        ]
        return self.status, self.output_files, self.output_values


# convertfort10mol
class Conversion_wf_workflow(Workflow):
    def __init__(
        self,
        conversion_wf_rerun: bool = False,
        conversion_wf_pkl_name: str = "conversion_wf_genius",
        to_wf: str = "agps",  # ['sd','agps', 'agpu', 'pf']
        grid_size: float = 0.10,
        additional_hyb: Optional[list] = None,
        nosym: bool = False,
        clean_flag: bool = True,
        only_generate_template: bool = False,
    ):
        if additional_hyb is None:
            additional_hyb = []
        # conversion_wf
        self.conversion_wf_rerun = conversion_wf_rerun
        self.conversion_wf_pkl_name = conversion_wf_pkl_name
        # variables
        self.to_wf = to_wf
        self.grid_size = grid_size
        self.additional_hyb = additional_hyb
        self.nosym = nosym
        self.clean_flag = clean_flag
        self.only_generate_template = only_generate_template
        self.wavefunction = Wavefunction()

        # return values
        self.status = "init"
        self.output_files = []
        self.output_values = {}

    async def async_launch(self):
        ###############################################
        # Start a workflow
        ###############################################
        self.root_dir = os.getcwd()
        logger.info(f"Current dir = {self.root_dir}")

        # ******************
        # conversion_wf
        # ******************
        os.chdir(self.root_dir)
        self.conversion_wf_dir = os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.conversion_wf_dir, "pkl")
        logger.info(f"Project root dir = {self.conversion_wf_dir}")
        self.conversion_wf_pkl = f"{self.conversion_wf_pkl_name}.pkl"

        if self.conversion_wf_rerun or not os.path.isfile(
            os.path.join(self.pkl_dir, self.conversion_wf_pkl)
        ):
            logger.info("Start: conversion_wf")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.conversion_wf_dir)

            # read WF
            self.wavefunction.read_from_fort10(fort10="fort.10")
            if self.to_wf == "sd":
                logger.error("Conversion to sd is not implemented yet.")
                raise NotImplementedError
            elif self.to_wf == "pf":
                logger.error("Conversion to pf is not implemented yet.")
                raise NotImplementedError
            elif self.to_wf in {"agps", "agpu"}:
                # singlet or triplet
                if self.to_wf == "agps":
                    triplet = False
                else:
                    triplet = True
                # WF conversion
                self.wavefunction.to_agp(
                    triplet=triplet,
                    pfaffian_flag=False,
                    grid_size=self.grid_size,
                    additional_hyb=self.additional_hyb,
                    nosym=self.nosym,
                    clean_flag=self.clean_flag,
                    only_generate_template=self.only_generate_template,
                )
            else:
                raise NotImplementedError

            with open(
                os.path.join(self.conversion_wf_dir, self.conversion_wf_pkl),
                "wb",
            ) as f:
                pickle.dump("dummy", f)
            with open(os.path.join(self.pkl_dir, self.conversion_wf_pkl), "wb") as f:
                pickle.dump("dummy", f)

            os.chdir(self.root_dir)

        else:
            logger.info("Skip: conversion_wf")

        logger.info("End: conversion_wf workflow ends.")
        await asyncio.sleep(1)
        os.chdir(self.root_dir)

        self.status = "success"
        p_list = [
            pathlib.Path(ob) for ob in glob.glob(os.path.join(self.root_dir, "*"))
        ]
        self.output_files = [
            str(p.resolve().relative_to(self.root_dir)) for p in p_list
        ]
        return self.status, self.output_files, self.output_values


if __name__ == "__main__":
    logger = getLogger("Turbo-Workflows")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
