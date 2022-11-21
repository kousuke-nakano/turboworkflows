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
from turbogenius.vmc_genius import VMC_genius

# pyturbo package
from turbogenius.pyturbo.io_fort10 import IO_fort10
from turbogenius.pyturbo.utils.utility import get_linenum_fort12

# jobmanager
from turbofilemanager.job_manager import Job_submission

class VMC_workflow(Workflow):
    def __init__(self,
        ## job
        server_machine_name="fugaku",
        cores=9216,
        openmp=1,
        queue="small",
        version="stable",
        sleep_time=1800, # sec.
        jobpkl_name="job_manager",
        ## vmc
        vmc_rerun=False,
        vmc_max_continuation=2,
        vmc_pkl_name="vmc_genius",
        vmc_target_error_bar=2.0e-5, # Ha
        vmc_trial_steps= 150,
        vmc_bin_block = 10,
        vmc_warmupblocks = 5,
        vmc_num_walkers = -1, # default -1 -> num of MPI process.
        vmc_twist_average=False,
        vmc_kpoints=[],
        vmc_force_calc_flag=False,
        vmc_maxtime=172000,
    ):
        #job
        self.server_machine_name = server_machine_name
        self.cores = cores
        self.openmp = openmp
        self.queue = queue
        self.version = version
        self.sleep_time = sleep_time
        self.jobpkl_name = jobpkl_name
        ## vmc
        self.vmc_rerun = vmc_rerun
        self.vmc_max_continuation = vmc_max_continuation
        self.vmc_pkl_name = vmc_pkl_name
        self.vmc_target_error_bar = vmc_target_error_bar
        self.vmc_trial_steps = vmc_trial_steps
        self.vmc_bin_block = vmc_bin_block
        self.vmc_warmupblocks = vmc_warmupblocks
        self.vmc_num_walkers = vmc_num_walkers
        self.vmc_twist_average = vmc_twist_average
        self.vmc_kpoints = vmc_kpoints
        self.vmc_force_calc_flag =vmc_force_calc_flag
        self.vmc_maxtime = vmc_maxtime
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
        #! VMC
        #******************
        os.chdir(self.root_dir)
        self.vmc_dir=os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.vmc_dir, "pkl")
        logger.info(f"Project root dir = {self.vmc_dir}")
        vmc_pkl_list=[f"{self.vmc_pkl_name}_{i}.pkl" for i in range(self.vmc_max_continuation)]

        #####
        # big loop for all the VMC continuations
        #####
        if self.vmc_rerun or not all([os.path.isfile(os.path.join(self.pkl_dir, vmc_pkl)) for vmc_pkl in vmc_pkl_list]):
            logger.info(f"Start: VMC calculation")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.vmc_dir)

            #####
            # continuation loop !! index is icont
            #####
            for icont in range(self.vmc_max_continuation):
                if icont==0:
                    logger.info(f"VMC test run, icont={icont}")
                elif icont==1:
                    logger.info(f"VMC initial run, icont={icont}")
                else:
                    logger.info(f"VMC continuation run, icont={icont}")

                self.vmc_pkl=f"{self.vmc_pkl_name}_{icont}.pkl"
                self.vmc_latest_pkl = f"{self.vmc_pkl_name}_latest.pkl"
                self.input_file = f"datasvmc_{icont}.input"
                self.output_file = f"out_vmc_{icont}"

                ####
                # run part
                ####
                if self.vmc_rerun or not os.path.isfile(os.path.join(self.vmc_dir, self.vmc_pkl)):
                    logger.info(f"{self.vmc_pkl} does not exist. or vmc_rerun = .true.")

                    if icont==0:
                        self.vmc_continuation_flag = False
                        logger.info(f"Run test for estimating steps for achieving the target error bar = {self.vmc_target_error_bar}")
                        ## estimated necesary steps per optimization to achieve the target error bar.
                        if self.vmc_trial_steps < 40 * self.vmc_bin_block + self.vmc_bin_block * self.vmc_warmupblocks:
                            logger.warning(f"vmcsteps = {self.vmc_trial_steps} is too small! < 40 * bin_block + bin_block * warmupblocks = {40 * self.vmc_bin_block + self.vmc_bin_block * self.vmc_warmupblocks}")
                            logger.warning(f"vmcsteps = {self.vmc_trial_steps} is set to 40 * bin_block + bin_block * warmupblocks = {40 * self.vmc_bin_block + self.vmc_bin_block * self.vmc_warmupblocks}")
                            self.vmc_trial_steps = 40 * self.vmc_bin_block + self.vmc_bin_block * self.vmc_warmupblocks
                        vmc_steps = self.vmc_trial_steps

                    else:
                        self.vmc_continuation_flag = True
                        pvmc_pkl=f"{self.vmc_pkl_name}_{icont-1}.pkl"
                        with open(os.path.join(self.vmc_dir, pvmc_pkl), "rb") as f:
                            vmc_genius = pickle.load(f)
                        mcmc_steps = get_linenum_fort12(os.path.join(self.vmc_dir, "fort.12"))
                        energy, error = vmc_genius.energy, vmc_genius.energy_error
                        logger.info(f"The error bar of the vmc energy {error:.5f} Ha per mcmc step={(mcmc_steps - self.vmc_bin_block * self.vmc_warmupblocks)}")
                        if error < self.vmc_target_error_bar:
                            logger.warning(f"The target error bar {self.vmc_target_error_bar} Ha has been already achieved!")
                            logger.warning(f"Exiting from the VMC continuation loop.")
                            break
                        vmc_steps_estimated_proper = int((mcmc_steps - self.vmc_bin_block * self.vmc_warmupblocks) * (error / self.vmc_target_error_bar) ** 2)
                        logger.info(f"The target error bar is {self.vmc_target_error_bar:.5f} Ha")
                        logger.info(f"The estimated steps to achieve the target error bar is {vmc_steps_estimated_proper:d} steps")

                        estimated_time_for_1_generation = vmc_genius.estimated_time_for_1_generation
                        estimated_time = estimated_time_for_1_generation * vmc_steps_estimated_proper
                        logger.info(f"Estimated time = {estimated_time:.0f} sec.")

                        vmc_steps = vmc_steps_estimated_proper

                    # generate a VMC instance
                    vmc_genius=VMC_genius(
                                     vmcsteps=vmc_steps,
                                     num_walkers=self.vmc_num_walkers,
                                     twist_average=self.vmc_twist_average,
                                     kpoints=self.vmc_kpoints,
                                     force_calc_flag=self.vmc_force_calc_flag,
                                     maxtime=self.vmc_maxtime
                                     )
                    # manual k points!!
                    #if len(self.vmc_kpoints) != 0 and self.vmc_twist_average == 2: vmc_genius.manual_kpoints=self.vmc_kpoints

                    vmc_genius.generate_input(input_name=self.input_file, cont=self.vmc_continuation_flag)

                    # binary set
                    if self.cores == self.openmp:
                        binary = "turborvb-serial.x"
                        nompi = True
                    else:
                        binary = "turborvb-mpi.x"
                        nompi = False

                    # Job submission by the job-manager package
                    job=Job_submission(local_machine_name="localhost",
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
                    job_submission_flag, job_number=job.job_submit(submission_script="submit.sh")
                    while not job_submission_flag:
                        logger.info(f"Waiting for submission")
                        #time.sleep(self.sleep_time)
                        await asyncio.sleep(self.sleep_time); os.chdir(self.vmc_dir)
                        job_submission_flag, job_number=job.job_submit(submission_script="submit.sh")
                    logger.info("Job submitted.")

                    with open(os.path.join(self.vmc_dir, self.vmc_pkl), "wb") as f:
                        pickle.dump(vmc_genius, f)

                else:
                    logger.info(f"{self.vmc_pkl} exists.")
                    with open(self.jobpkl, "rb") as f:
                        job=pickle.load(f)
                    with open(os.path.join(self.vmc_dir, self.vmc_pkl), "rb") as f:
                        vmc_genius=pickle.load(f)

                ####
                # Fetch part
                ####
                if self.vmc_rerun or not os.path.isfile(os.path.join(self.pkl_dir, self.vmc_pkl)):
                    logger.info(f"{self.vmc_pkl} does not exist in {self.pkl_dir}.")
                    logger.info("job is running or fetch has not been done yet.")
                    # job waiting
                    job_running = job.jobcheck()
                    while job_running:
                        logger.info(f"Waiting for the submitted job = {job.job_number}");
                        #time.sleep(self.sleep_time)
                        await asyncio.sleep(self.sleep_time); os.chdir(self.vmc_dir)
                        job_running = job.jobcheck()
                    logger.info("Job finished.")
                    # job fetch
                    logger.info("Fetch files.")
                    fetch_files=[self.output_file, "fort.11", "fort.12", "parminimized.d"]
                    exclude_files = []
                    if self.vmc_twist_average:
                        fetch_files += ["kp_info.dat", "turborvb.scratch"]
                        exclude_files += ["kelcont*", "randseed*"]
                    job.fetch_job(from_objects=fetch_files, exclude_list=exclude_files)
                    logger.info("Fetch finished.")

                    logger.info(f"Computing VMC forces")
                    vmc_genius.compute_energy_and_forces(bin_block=self.vmc_bin_block, warmupblocks=self.vmc_warmupblocks, rerun=True)
                    vmc_genius.store_result(bin_block=self.vmc_bin_block, warmupblocks=self.vmc_warmupblocks, output_names=[f"out_vmc_{i}" for i in range(icont+1)])
                    energy, error = vmc_genius.energy, vmc_genius.energy_error
                    estimated_time_for_1_generation = vmc_genius.estimated_time_for_1_generation
                    logger.info(f"VMC energy = {energy:.5f} +- {error:3f} Ha")
                    logger.info(f"estimated_time_for_1_generation = {estimated_time_for_1_generation:.5f} sec")

                    self.output_values["energy"] = energy
                    self.output_values["error"] = error

                    with open(os.path.join(self.vmc_dir, self.vmc_pkl), "wb") as f:
                        pickle.dump(vmc_genius, f)
                    with open(os.path.join(self.pkl_dir, self.vmc_pkl), "wb") as f:
                        pickle.dump(vmc_genius, f)
                    with open(os.path.join(self.pkl_dir, self.vmc_latest_pkl), "wb") as f:
                        pickle.dump(vmc_genius, f)

                logger.info(f"VMC run ends for icont={icont}")

            os.chdir(self.root_dir)

        else:
            logger.info(f"Skip: VMC calculation")
            self.vmc_latest_pkl = f"{self.vmc_pkl_name}_latest.pkl"
            with open(os.path.join(self.pkl_dir,  self.vmc_latest_pkl), "rb") as f:
                vmc_genius=pickle.load(f)
            energy, error = vmc_genius.energy, vmc_genius.energy_error
            logger.info(f"VMC energy = {energy:.5f} +- {error:3f} Ha")
            self.output_values["energy"] = energy
            self.output_values["error"] = error

        logger.info("VMC workflow ends.")
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