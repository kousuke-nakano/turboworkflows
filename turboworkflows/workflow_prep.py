#!/usr/bin/env python
# coding: utf-8

# python packages
import numpy as np
import os, sys
import shutil
import pickle
import numpy as np
import time
import glob
import asyncio
import pathlib

#Logger
from logging import config, getLogger, StreamHandler, Formatter, FileHandler
logger = getLogger('Turbo-Workflows').getChild(__name__)

# turboworkflow packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from workflow_encapsulated import Workflow
from utils_turboworkflows.turboworkflows_env import turbo_workflows_root

# turbo-genius packages
from turbogenius.prep_genius import DFT_genius

# pyturbo package
from turbogenius.pyturbo.io_fort10 import IO_fort10
from turbogenius.pyturbo.utils.utility import get_linenum_fort12

# jobmanager
from turbofilemanager.job_manager import Job_submission

class DFT_workflow(Workflow):
    def __init__(self,
        ## job
        server_machine_name="fugaku",
        cores=9216,
        openmp=1,
        queue="small",
        version="stable",
        sleep_time=1800, # sec.
        jobpkl_name="job_manager",
        ## prep
        dft_rerun = False,
        dft_pkl_name = "prep",
        dft_grid_size=[0.1, 0.1, 0.1],
        dft_lbox=[15.0, 15.0, 15.0],
        dft_smearing=0.0,
        dft_maxtime=172800,
        dft_memlarge=False,
        dft_h_field=0.0,
        dft_magnetic_moment_list=[],
        dft_xc='lda',  # lda or lsda
        dft_twist_average=False,
        dft_kpoints=[1, 1, 1, 0, 0, 0]
    ):
        #job
        self.server_machine_name = server_machine_name
        self.cores = cores
        self.openmp = openmp
        self.queue = queue
        self.version = version
        self.sleep_time = sleep_time
        self.jobpkl_name = jobpkl_name
        ## dft
        self.dft_rerun = dft_rerun
        self.dft_pkl_name = dft_pkl_name
        self.dft_grid_size=dft_grid_size
        self.dft_lbox=dft_lbox
        self.dft_smearing=dft_smearing
        self.dft_maxtime=dft_maxtime
        self.dft_memlarge=dft_memlarge
        self.dft_h_field=dft_h_field
        self.dft_magnetic_moment_list=dft_magnetic_moment_list
        self.dft_xc=dft_xc
        self.dft_twist_average=dft_twist_average
        self.dft_kpoints=dft_kpoints
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
        #! dft
        #******************
        os.chdir(self.root_dir)
        self.dft_dir=os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.dft_dir, "pkl")
        logger.info(f"Project root dir = {self.dft_dir}")
        self.dft_pkl = f"{self.dft_pkl_name}.pkl"

        if self.dft_rerun or not os.path.isfile(os.path.join(self.pkl_dir, self.dft_pkl)):
            logger.info(f"Start: DFT calculation")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.dft_dir)

            self.input_file = f"prep.input"
            self.output_file = f"out_prep"

            ####
            # run part
            ####
            if self.dft_rerun or not os.path.isfile(os.path.join(self.dft_dir, self.dft_pkl)):
                logger.info(f"{self.dft_pkl} does not exist. or dft_rerun = .true.")

                # generate a DFT instance
                dft_genius=DFT_genius(
                                 grid_size=self.dft_grid_size,
                                 lbox=self.dft_lbox,
                                 smearing=self.dft_smearing,
                                 maxtime=self.dft_maxtime,
                                 memlarge=self.dft_memlarge,
                                 h_field=self.dft_h_field,
                                 magnetic_moment_list=self.dft_magnetic_moment_list,
                                 xc=self.dft_xc,
                                 twist_average=self.dft_twist_average,
                                 kpoints=self.dft_kpoints
                        )

                dft_genius.generate_input(input_name=self.input_file)

                # Job submission by the job-manager package
                job=Job_submission(local_machine_name="localhost",
                                   client_machine_name="localhost",
                                   server_machine_name=self.server_machine_name,
                                   package="turborvb",
                                   cores=self.cores,
                                   openmp=self.openmp,
                                   queue=self.queue,
                                   version=self.version,
                                   binary="prep-mpi.x",
                                   jobname="turbogenius",
                                   input_file=self.input_file,
                                   output_file=self.output_file,
                                   pkl_name=self.jobpkl
                                   )
                job.generate_script(submission_script="submit.sh")
                # job submission
                job_submission_flag, job_number=job.job_submit(submission_script="submit.sh")
                while not job_submission_flag:
                    logger.info(f"Waiting for submission")
                    #time.sleep(self.sleep_time)
                    await asyncio.sleep(self.sleep_time); os.chdir(self.dft_dir)
                    job_submission_flag, job_number=job.job_submit(submission_script="submit.sh")
                logger.info("Job submitted.")

                with open(os.path.join(self.dft_dir, self.dft_pkl), "wb") as f:
                    pickle.dump(dft_genius, f)

            else:
                logger.info(f"{self.dft_pkl} exists.")
                with open(self.jobpkl, "rb") as f:
                    job=pickle.load(f)
                with open(os.path.join(self.dft_dir, self.dft_pkl), "rb") as f:
                    dft_genius=pickle.load(f)

            ####
            # Fetch part
            ####
            if self.dft_rerun or not os.path.isfile(os.path.join(self.pkl_dir, self.dft_pkl)):
                logger.info(f"{self.dft_pkl} does not exist in {self.pkl_dir}.")
                logger.info("job is running or fetch has not been done yet.")
                # job waiting
                job_running = job.jobcheck()
                while job_running:
                    logger.info(f"Waiting for the submitted job = {job.job_number}");
                    #time.sleep(self.sleep_time)
                    await asyncio.sleep(self.sleep_time); os.chdir(self.dft_dir)
                    job_running = job.jobcheck()
                logger.info("Job finished.")
                # job fecth
                logger.info("Fetch files.")
                fetch_files=[self.output_file, "occupationlevels.dat", "EDFT_vsk.dat"]
                exclude_files=[]
                if self.dft_twist_average:
                    fetch_files+=["kp_info.dat", "turborvb.scratch"]
                    exclude_files+=["kelcont*", "randseed*"]
                else:
                    fetch_files+=["fort.10_new"]
                job.fetch_job(from_objects=fetch_files, exclude_list=exclude_files)
                logger.info("Fetch finished.")

                self.output_values["energy"] = None
                self.output_values["error"] = None

                with open(os.path.join(self.dft_dir, self.dft_pkl), "wb") as f:
                    pickle.dump(dft_genius, f)
                with open(os.path.join(self.pkl_dir, self.dft_pkl), "wb") as f:
                    pickle.dump(dft_genius, f)

            os.chdir(self.root_dir)

        else:
            logger.info(f"Skip: DFT calculation")
            self.dft_pkl = f"{self.dft_pkl_name}.pkl"
            with open(os.path.join(self.pkl_dir,  self.dft_pkl), "rb") as f:
                dft_genius=pickle.load(f)
            energy, error = None, None
            self.output_values["energy"] = energy
            self.output_values["error"] = error

        logger.info("DFT workflow ends.")
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

    os.chdir(os.path.join(turbo_workflows_root, "tests", "prep-workflows"))

    prep_workflow=DFT_workflow(
        ## job
        server_machine_name="kagayaki",
        cores=64,
        openmp=1,
        queue="DEFAULT",
        version="stable",
        sleep_time=30, # sec.
        jobpkl_name="job_manager",
        ## prep
        dft_rerun = False,
        dft_pkl_name = "prep",
        dft_grid_size=[0.1, 0.1, 0.1],
        dft_lbox=[15.0, 15.0, 15.0],
        dft_smearing=0.0,
        dft_maxtime=172800,
        dft_h_field=0.0,
        dft_magnetic_moment_list=[],
        dft_xc='lda',  # lda or lsda
        dft_twist_average=True,
        dft_kpoints=[2, 2, 2, 0, 0, 0]
    )

    prep_workflow.launch()
    # moved to examples
