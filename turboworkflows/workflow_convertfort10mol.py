#!/usr/bin/env python
# coding: utf-8

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

# turboworkflows packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from workflow_encapsulated import Workflow
from utils_turboworkflows.turboworkflows_env import turbo_workflows_root

# turbo-genius packages
from turbogenius.convertfort10mol_genius import Convertfort10mol_genius

# pyturbo package
from pyturbo.io_fort10 import IO_fort10

# jobmanager
from turbofilemanager.job_manager import Job_submission

# convertfort10mol
class Convertfort10mol_workflow(Workflow):
    def __init__(self,
                 ## job
                 server_machine_name="fugaku",
                 cores=9216,
                 openmp=1,
                 queue="small",
                 version="stable",
                 sleep_time=1800,  # sec.
                 jobpkl_name="job_manager",
                 #convertfort10mol
                 convertfort10mol_rerun=False,
                 convertfort10mol_pkl_name="convertfort10mol_genius",
                 add_random_mo=True,
                 grid_size=0.10,
                 additional_mo=0,
                 ):

        #job
        self.server_machine_name = server_machine_name
        self.cores = cores
        self.openmp = openmp
        self.queue = queue
        self.version = version
        self.sleep_time = sleep_time
        self.jobpkl_name = jobpkl_name
        ##convertfort10mol
        self.convertfort10mol_rerun = convertfort10mol_rerun
        self.convertfort10mol_pkl_name = convertfort10mol_pkl_name
        self.add_random_mo = add_random_mo
        self.grid_size = grid_size
        self.additional_mo = additional_mo
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
        #! convertfort10mol
        #******************
        os.chdir(self.root_dir)
        self.convertfort10mol_dir=os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.convertfort10mol_dir, "pkl")
        logger.info(f"Project root dir = {self.convertfort10mol_dir}")
        self.convertfort10mol_pkl= f"{self.convertfort10mol_pkl_name}.pkl"

        if self.convertfort10mol_rerun or not os.path.isfile(os.path.join(self.pkl_dir, self.convertfort10mol_pkl)):
            logger.info(f"Start: convertfort10mol")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.convertfort10mol_dir)
            self.input_file = f"convertfort10mol.input"
            self.output_file = f"out_mol"

            if self.convertfort10mol_rerun or not os.path.isfile(os.path.join(self.convertfort10mol_dir, self.convertfort10mol_pkl)):

                convertfort10mol_genius=Convertfort10mol_genius(
                                 add_random_mo=self.add_random_mo,
                                 grid_size=self.grid_size,
                                 additional_mo=self.additional_mo,
                                 )

                convertfort10mol_genius.generate_input(input_name=self.input_file)

                # binary set
                if self.cores==self.openmp:
                    binary="convertfort10mol.x"
                    nompi = True
                else:
                    binary="convertfort10mol-mpi.x"
                    nompi = False

                # Job submission by the job-manager package
                job = Job_submission(local_machine_name="localhost",
                                     client_machine_name="localhost",
                                     server_machine_name=self.server_machine_name,
                                     package="turborvb",
                                     cores=self.cores,
                                     openmp=self.openmp,
                                     queue=self.queue,
                                     version=self.version,
                                     binary=binary,
                                     nompi=nompi,
                                     jobname="turbogenius",
                                     input_file=self.input_file,
                                     output_file=self.output_file,
                                     pkl_name=self.jobpkl
                                     )
                job.generate_script(submission_script="submit.sh")
                # job submission
                job_submission_flag, job_number = job.job_submit(submission_script="submit.sh")
                while not job_submission_flag:
                    logger.info(f"Waiting for submission")
                    # time.sleep(self.sleep_time)
                    await asyncio.sleep(self.sleep_time);
                    os.chdir(self.convertfort10mol_dir)
                    job_submission_flag, job_number = job.job_submit(submission_script="submit.sh")
                logger.info("Job submitted.")

                with open(os.path.join(self.convertfort10mol_dir, self.convertfort10mol_pkl), "wb") as f:
                    pickle.dump(convertfort10mol_genius, f)

            else:
                logger.info(f"{self.convertfort10mol_pkl} exists.")
                with open(self.jobpkl, "rb") as f:
                    job = pickle.load(f)
                with open(os.path.join(self.convertfort10mol_dir, self.convertfort10mol_pkl), "rb") as f:
                    convertfort10mol_genius = pickle.load(f)

            ####
            # Fetch part
            ####
            if self.convertfort10mol_rerun or not os.path.isfile(os.path.join(self.pkl_dir, self.convertfort10mol_pkl)):
                logger.info(f"{self.convertfort10mol_pkl} does not exist in {self.pkl_dir}.")
                logger.info("job is running or fetch has not been done yet.")
                # job waiting
                job_running = job.jobcheck()
                while job_running:
                    logger.info(f"Waiting for the submitted job = {job.job_number}");
                    # time.sleep(self.sleep_time)
                    await asyncio.sleep(self.sleep_time);
                    os.chdir(self.convertfort10mol_dir)
                    job_running = job.jobcheck()
                logger.info("Job finished.")
                # job fecth
                logger.info("Fetch files.")
                fetch_files = [self.output_file, "fort.10_new"]
                exclude_files = []
                job.fetch_job(from_objects=fetch_files, exclude_list=exclude_files)
                logger.info("Fetch finished.")

                with open(os.path.join(self.convertfort10mol_dir, self.convertfort10mol_pkl), "wb") as f:
                    pickle.dump(convertfort10mol_genius, f)
                with open(os.path.join(self.pkl_dir, self.convertfort10mol_pkl), "wb") as f:
                    pickle.dump(convertfort10mol_genius, f)

                os.chdir(self.root_dir)

        else:
            logger.info(f"Skip: convertfort10mol")
            self.convertfort10mol_pkl = f"{self.convertfort10mol_pkl_name}.pkl"
            with open(os.path.join(self.pkl_dir, self.convertfort10mol_pkl), "rb") as f:
                convertfort10mol_genius = pickle.load(f)

        logger.info("End: convertfort10mol workflow ends.")
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

    # moved to examples