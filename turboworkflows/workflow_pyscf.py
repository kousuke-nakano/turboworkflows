#!/usr/bin/env python
# coding: utf-8

# python packages
import numpy as np
import os, sys
import shutil
import pickle
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

# pyturbo package
from turbogenius.pyturbo.io_fort10 import IO_fort10

# pyscf package
from pyscf import gto, scf, mp, tools

# jobmanager
from turbofilemanager.job_manager import Job_submission

class PySCF_workflow(Workflow):
    def __init__(self,
        ## structure file (mandatory)
        structure_file,
        trexio_filename="trexio.hdf5",
        ## job
        server_machine_name="fugaku",
        cores=9216,
        openmp=1,
        queue="small",
        version="stable",
        sleep_time=1800, # sec.
        jobpkl_name="job_manager",
        ## pyscf
        pyscf_rerun=False,
        pyscf_pkl_name="pyscf_genius",
        init_guess = 'minao',
        charge=0,
        spin=0,
        basis="ccecp-ccpvtz",  # defined below
        ecp="ccecp",  # defined below
        scf_method="DFT",  # HF or DFT
        dft_xc="LDA_X,LDA_C_PZ",
        mp2_flag=False,
        ccsd_flag=False,
        pyscf_output="out.pyscf",
        pyscf_chkfile="pyscf.chk",
        solver_newton=False,
        twist_average=False,
        exp_to_discard=0.10,
        kpt=[0.0, 0.0, 0.0],  # scaled_kpts!! i.e., crystal coord.
        kpt_grid=[1, 1, 1],
        smearing_method="fermi",
        smearing_sigma=0.00,  # Ha
        ## conversion to trexio file
        force_wf_complex = False
        ):

        #structure
        self.structure_file = structure_file
        self.trexio_filename=trexio_filename
        #job
        self.server_machine_name = server_machine_name
        self.cores = cores
        self.openmp = openmp
        self.queue = queue
        self.version = version
        self.sleep_time = sleep_time
        self.jobpkl_name = jobpkl_name
        #pyscf
        self.pyscf_rerun=pyscf_rerun
        self.pyscf_pkl_name=pyscf_pkl_name
        self.init_guess=init_guess
        self.charge=charge
        self.spin=spin
        self.basis=basis  # defined below
        self.ecp=ecp  # defined below
        self.scf_method=scf_method  # HF or DFT
        self.dft_xc=dft_xc
        self.mp2_flag=mp2_flag
        self.ccsd_flag=ccsd_flag
        self.pyscf_output=pyscf_output
        self.pyscf_chkfile=pyscf_chkfile
        self.solver_newton=solver_newton
        self.twist_average=twist_average
        self.exp_to_discard=exp_to_discard
        self.kpt=kpt  # scaled_kpts!! i.e., crystal coord.
        self.kpt_grid=kpt_grid
        self.smearing_method = smearing_method
        self.smearing_sigma = smearing_sigma
        #conversion to trexio file
        self.force_wf_complex = force_wf_complex
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
        self.jobpkl = f"{self.jobpkl_name}.pkl"

        #******************
        #! pyscf
        #******************
        os.chdir(self.root_dir)
        self.pyscf_dir=os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.pyscf_dir, "pkl")
        self.pyscf_pkl = f"{self.pyscf_pkl_name}.pkl"
        logger.info(f"Project root dir = {self.pyscf_dir}")

        if self.pyscf_rerun or not os.path.isfile(os.path.join(self.pkl_dir, self.pyscf_pkl)):
            logger.info(f"Start: pyscf calculation")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.pyscf_dir)

            ####
            # run part
            ####
            if self.pyscf_rerun or not os.path.isfile(os.path.join(self.pyscf_dir, self.pyscf_pkl)):
                logger.info(f"{self.pyscf_pkl} does not exist. or pyscf_rerun = .true.")

                pyscf_python_wrapper=os.path.join(turbo_workflows_source_root, "pyscf_tools", "pyscf_wrapper.py")
                shutil.copy(pyscf_python_wrapper, self.pyscf_dir)

                def rg(arg):
                    if type(arg) == str:
                        return '\"' + arg + '\"'
                    else:
                        return arg

                run_py=f"""
from pyscf_wrapper import Pyscf_wrapper

# input variables
pyscf_chkfile={rg(self.pyscf_chkfile)}
structure_file={rg(self.structure_file)}

# input variables
omp_num_threads={rg(self.openmp)}
init_guess={rg(self.init_guess)}
charge={rg(self.charge)}
spin={rg(self.spin)}
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

pyscf_calc=Pyscf_wrapper(
                        structure_file=structure_file,
                        chkfile=pyscf_chkfile,
                        )

pyscf_calc.run_pyscf(
                  omp_num_threads=omp_num_threads,
                  init_guess=init_guess,
                  charge=charge,
                  spin=spin,
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
                  smearing_sigma=smearing_sigma
                  )
                """

                with open(os.path.join(self.pyscf_dir,"run.py"), "w") as f:
                    f.write(run_py)

                job = Job_submission(local_machine_name="localhost",
                                     client_machine_name="localhost",
                                     server_machine_name=self.server_machine_name,
                                     package="python",
                                     cores=self.cores,
                                     openmp=self.openmp,
                                     queue=self.queue,
                                     version=self.version,
                                     jobname="pyscf",
                                     input_file="run.py",
                                     nompi=True,
                                     input_redirect=False,
                                     pkl_name=self.jobpkl
                                     )
                job.generate_script(submission_script="submit.sh")

                # job submission
                job_submission_flag, job_number = job.job_submit(submission_script="submit.sh")
                while not job_submission_flag:
                    logger.info(f"Waiting for submission");
                    #time.sleep(self.sleep_time)
                    await asyncio.sleep(self.sleep_time); os.chdir(self.pyscf_dir)
                    job_submission_flag, job_number = job.job_submit(submission_script="submit.sh")
                logger.info("Job submitted.")

                with open(os.path.join(self.pyscf_dir, self.pyscf_pkl), "wb") as f:
                    pickle.dump("dummy", f)

            else:
                logger.info(f"{self.pyscf_pkl} exists.")
                with open(self.jobpkl, "rb") as f:
                    job = pickle.load(f)

            ####
            # Fetch part
            ####
            if self.pyscf_rerun or not os.path.isfile(os.path.join(self.pkl_dir, self.pyscf_pkl)):
                logger.info(f"{self.pyscf_pkl} does not exist in {self.pkl_dir}.")
                logger.info("job is running or fetch has not been done yet.")
                # job waiting
                job_running = job.jobcheck()
                while job_running:
                    logger.info(f"Waiting for the submitted job = {job.job_number}");
                    #time.sleep(self.sleep_time)
                    await asyncio.sleep(self.sleep_time); os.chdir(self.pyscf_dir)
                    job_running = job.jobcheck()
                logger.info("Job finished.")
                # job fecth
                logger.info("Fetch files.")
                fetch_files = [self.pyscf_output, self.pyscf_chkfile]
                job.fetch_job(from_objects=fetch_files)
                logger.info("Fetch finished.")

                mf = scf.chkfile.load(os.path.join(self.pyscf_dir, self.pyscf_chkfile), "scf")
                energy = mf['e_tot']
                self.output_values["energy"] = energy
                logger.info(f"PySCF energy = {energy}")

                ####
                # conversion to the TREXIO format
                ####
                logger.info(f"Start: pyscf -> trexio conversion.")
                pyscf_to_trexio(pyscf_checkfile=self.pyscf_chkfile,
                                trexio_filename=os.path.join(self.pyscf_dir, self.trexio_filename),
                                twist_average_in=self.twist_average,
                                force_wf_complex=self.force_wf_complex
                                )
                logger.info(f"End: pyscf -> trexio conversion.")

            with open(os.path.join(self.pyscf_dir, self.pyscf_pkl), "wb") as f:
                pickle.dump("dummy", f)
            with open(os.path.join(self.pkl_dir, self.pyscf_pkl), "wb") as f:
                pickle.dump("dummy", f)

        else:
            mf = scf.chkfile.load(os.path.join(self.pyscf_dir,self.pyscf_chkfile), "scf")
            energy = mf['e_tot']
            self.output_values["energy"] = energy
            logger.info(f"PySCF energy = {energy}")

        logger.info("pySCF workflow ends.")
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
