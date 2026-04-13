#!/usr/bin/env python
# coding: utf-8

# python packages
import os
import shutil
import pickle
import asyncio
import glob
import pathlib
from typing import Optional, Union

# Logger
from logging import getLogger, StreamHandler, Formatter

# turboworkflow packages
from .turbofilemanager.job_manager import Job_submission
from .workflow_encapsulated import Workflow
from .utils_turboworkflows.turboworkflows_env import (
    turbo_workflows_source_root,
)

# pyscf package
from pyscf import scf

logger = getLogger("Turbo-Workflows").getChild(__name__)


class PySCF_workflow(Workflow):
    def __init__(
        self,
        # structure file (mandatory)
        structure_file: str,
        ghost_atoms_index: Optional[list] = None,
        trexio_filename: str = "trexio.hdf5",
        # job
        server_machine_name: str = "localhost",
        queue_label: Optional[str] = "default",
        mpi=False,
        version: str = "stable",
        sleep_time: int = 1800,  # sec.
        # pyscf
        pyscf_rerun: bool = False,
        init_guess: str = "minao",
        cell_precision: float = 1.0e-8,
        multigrid_fftdf: bool = False,
        level_shift_factor: float = 0.0,
        charge: int = 0,
        spin: int = 0,
        spin_restricted: bool = True,
        basis: Union[str, dict] = "ccecp-ccpvtz",  # defined below
        ecp: Union[str, dict] = "ccecp",  # defined below
        scf_method: str = "DFT",  # HF or DFT
        dft_xc: str = "LDA_X,LDA_C_PZ",
        mp2_flag: bool = False,
        ccsd_flag: bool = False,
        pyscf_output: str = "out.pyscf",
        pyscf_chkfile: str = "pyscf.chk",
        solver_newton: bool = False,
        twist_average: bool = False,
        exp_to_discard: float = 0.10,
        kpt: Optional[list] = None,  # scaled_kpts!! i.e., crystal coord.
        kpt_grid: Optional[list] = None,
        smearing_method: str = "fermi",
        smearing_sigma: float = 0.00,  # Ha
        # conversion to trexio file
        force_wf_complex: bool = False,
        use_jkmethod: bool = False,
    ):
        if ghost_atoms_index is None:
            ghost_atoms_index = []
        if kpt is None:
            kpt = [0.0, 0.0, 0.0]
        if kpt_grid is None:
            kpt_grid = [1, 1, 1]

        # structure
        self.structure_file = structure_file
        self.ghost_atoms_index = ghost_atoms_index
        self.trexio_filename = trexio_filename
        # job
        self.server_machine_name = server_machine_name
        self.queue_label = queue_label
        self.mpi = mpi
        self.version = version
        self.sleep_time = sleep_time
        # pyscf
        self.pyscf_rerun = pyscf_rerun
        self.init_guess = init_guess
        self.cell_precision = cell_precision
        self.multigrid_fftdf = multigrid_fftdf
        self.level_shift_factor = level_shift_factor
        self.charge = charge
        self.spin = spin
        self.spin_restricted = spin_restricted
        self.basis = basis  # defined below
        self.ecp = ecp  # defined below
        self.scf_method = scf_method  # HF or DFT
        self.dft_xc = dft_xc
        self.mp2_flag = mp2_flag
        self.ccsd_flag = ccsd_flag
        self.pyscf_output = pyscf_output
        self.pyscf_chkfile = pyscf_chkfile
        self.solver_newton = solver_newton
        self.twist_average = twist_average
        self.exp_to_discard = exp_to_discard
        self.kpt = kpt  # scaled_kpts!! i.e., crystal coord.
        self.kpt_grid = kpt_grid
        self.smearing_method = smearing_method
        self.smearing_sigma = smearing_sigma
        self.use_jkmethod = use_jkmethod
        # conversion to trexio file
        self.force_wf_complex = force_wf_complex
        # pkl names
        self.job_pkl_name = "job_manager"
        self.pyscf_pkl_name = "pyscf_genius"
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
        self.job_pkl = f"{self.job_pkl_name}.pkl"

        # ******************
        # pyscf
        # ******************
        os.chdir(self.root_dir)
        self.pyscf_dir = os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.pyscf_dir, "pkl")
        self.pyscf_pkl = f"{self.pyscf_pkl_name}.pkl"
        logger.info(f"Project root dir = {self.pyscf_dir}")

        if self.pyscf_rerun or not os.path.isfile(
            os.path.join(self.pkl_dir, self.pyscf_pkl)
        ):
            logger.info("Start: pyscf calculation")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.pyscf_dir)

            ####
            # run part
            ####
            if self.pyscf_rerun or not os.path.isfile(
                os.path.join(self.pyscf_dir, self.pyscf_pkl)
            ):
                logger.info(f"{self.pyscf_pkl} does not exist. or pyscf_rerun = .true.")

                pyscf_python_wrapper = os.path.join(
                    turbo_workflows_source_root,
                    "pyscf_tools",
                    "pyscf_wrapper.py",
                )
                shutil.copy(pyscf_python_wrapper, self.pyscf_dir)

                def rg(arg):
                    if type(arg) is str:
                        return '"' + arg + '"'
                    else:
                        return arg

                run_py = f"""
from pyscf_wrapper import run_pyscf

# input variables
pyscf_chkfile={rg(self.pyscf_chkfile)}
trexio_filename={rg(self.trexio_filename)}
structure_file={rg(self.structure_file)}
ghost_atoms_index={rg(self.ghost_atoms_index)}

# input variables
init_guess={rg(self.init_guess)}
cell_precision={rg(self.cell_precision)}
multigrid_fftdf={rg(self.multigrid_fftdf)}
level_shift_factor={rg(self.level_shift_factor)}
charge={rg(self.charge)}
spin={rg(self.spin)}
spin_restricted={rg(self.spin_restricted)}
basis={rg(self.basis)}
ecp={rg(self.ecp)}
scf_method={rg(self.scf_method)}
dft_xc={rg(self.dft_xc)}
solver_newton={rg(self.solver_newton)}
MP2_flag={rg(self.mp2_flag)}
CCSD_flag={rg(self.ccsd_flag)}
pyscf_output={rg(self.pyscf_output)}
twist_average={rg(self.twist_average)}
exp_to_discard={rg(self.exp_to_discard)}
kpt={rg(self.kpt)}
kpt_grid={rg(self.kpt_grid)}
smearing_method={rg(self.smearing_method)}
smearing_sigma={rg(self.smearing_sigma)}
use_jkmethod={rg(self.use_jkmethod)}

run_pyscf(
        structure_file=structure_file,
        ghost_atoms_index=ghost_atoms_index,
        chkfile=pyscf_chkfile,
        trexio_filename=trexio_filename,
        init_guess=init_guess,
        cell_precision=cell_precision,
        multigrid_fftdf=multigrid_fftdf,
        level_shift_factor=level_shift_factor,
        charge=charge,
        spin=spin,
        spin_restricted=spin_restricted,
        basis=basis,
        ecp=ecp,
        scf_method=scf_method,
        dft_xc=dft_xc,
        solver_newton=solver_newton,
        MP2_flag=MP2_flag,
        CCSD_flag=CCSD_flag,
        pyscf_output=pyscf_output,
        twist_average=twist_average,
        exp_to_discard=exp_to_discard,
        kpt=kpt,
        kpt_grid=kpt_grid,
        smearing_method=smearing_method,
        smearing_sigma=smearing_sigma,
        use_jkmethod=use_jkmethod
        )
                """

                with open(os.path.join(self.pyscf_dir, "run.py"), "w") as f:
                    f.write(run_py)

                # binary set
                if self.mpi:
                    logger.error("mpi=True is not implemented for pyscf workflows.")
                    logger.error("openmp is supported.")
                    raise NotImplementedError

                job = Job_submission(
                    client_machine_name="localhost",
                    server_machine_name=self.server_machine_name,
                    package="python",
                    queue_label=self.queue_label,
                    version=self.version,
                    mpi=False,
                    jobname="pyscf",
                    input_file="run.py",
                    input_redirect=False,
                    pkl_name=self.job_pkl,
                )
                job.generate_script(submission_script="submit.sh")

                # job submission
                job_submission_flag, job_number = job.job_submit(
                    submission_script="submit.sh"
                )
                while not job_submission_flag:
                    logger.info("Waiting for submission")
                    # time.sleep(self.sleep_time)
                    await asyncio.sleep(self.sleep_time)
                    os.chdir(self.pyscf_dir)
                    job_submission_flag, job_number = job.job_submit(
                        submission_script="submit.sh"
                    )
                logger.info("Job submitted.")

                with open(os.path.join(self.pyscf_dir, self.pyscf_pkl), "wb") as f:
                    pickle.dump("dummy", f)

            else:
                logger.info(f"{self.pyscf_pkl} exists.")
                with open(self.job_pkl, "rb") as f:
                    job = pickle.load(f)

            ####
            # Fetch part
            ####
            if self.pyscf_rerun or not os.path.isfile(
                os.path.join(self.pkl_dir, self.pyscf_pkl)
            ):
                logger.info(f"{self.pyscf_pkl} does not exist in {self.pkl_dir}.")
                logger.info("job is running or fetch has not been done yet.")
                # job waiting
                job_running = job.jobcheck()
                while job_running:
                    logger.info(f"Waiting for the submitted job = {job.job_number}")
                    # time.sleep(self.sleep_time)
                    await asyncio.sleep(self.sleep_time)
                    os.chdir(self.pyscf_dir)
                    job_running = job.jobcheck()
                logger.info("Job finished.")
                # job fecth
                logger.info("Fetch files.")
                fetch_files = [self.pyscf_output, self.pyscf_chkfile, self.trexio_filename, "int1e_ovlp.npy"]
                job.fetch_job(from_objects=fetch_files)
                logger.info("Fetch finished.")

                mf = scf.chkfile.load(
                    os.path.join(self.pyscf_dir, self.pyscf_chkfile), "scf"
                )
                energy = mf["e_tot"]
                self.output_values["energy"] = energy
                logger.info(f"PySCF energy = {energy}")

                # TREXIO file is now generated directly by pyscf-forge in run_pyscf()

            with open(os.path.join(self.pyscf_dir, self.pyscf_pkl), "wb") as f:
                pickle.dump("dummy", f)
            with open(os.path.join(self.pkl_dir, self.pyscf_pkl), "wb") as f:
                pickle.dump("dummy", f)

        else:
            mf = scf.chkfile.load(
                os.path.join(self.pyscf_dir, self.pyscf_chkfile), "scf"
            )
            energy = mf["e_tot"]
            self.output_values["energy"] = energy
            logger.info(f"PySCF energy = {energy}")

        logger.info("pySCF workflow ends.")
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
