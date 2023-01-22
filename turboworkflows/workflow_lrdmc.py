#!/usr/bin/env python
# coding: utf-8

# python packages
import os
import pickle
import asyncio
import glob
import pathlib

# Logger
from logging import getLogger, StreamHandler, Formatter

# turboworkflows packages
from turboworkflows.workflow_encapsulated import Workflow

# turbo-genius packages
from turbogenius.lrdmc_genius import LRDMC_genius

# pyturbo package
from turbogenius.pyturbo.utils.utility import get_linenum_fort12

# jobmanager
from turbofilemanager.job_manager import Job_submission

logger = getLogger("Turbo-Workflows").getChild(__name__)


class LRDMC_workflow(Workflow):
    def __init__(
        self,
        # job
        server_machine_name="fugaku",
        cores=9216,
        openmp=1,
        queue="small",
        version="stable",
        sleep_time=1800,  # sec.
        jobpkl_name="job_manager",
        # lrdmc
        lrdmc_rerun=False,
        lrdmc_max_continuation=2,
        lrdmc_pkl_name="lrdmc_genius",
        lrdmc_target_error_bar=2.0e-5,  # Ha
        lrdmc_trial_steps=150,
        lrdmc_bin_block=10,
        lrdmc_warmupblocks=5,
        lrdmc_safe_trial_steps=True,
        lrdmc_correcting_factor=10,
        lrdmc_trial_etry=0.0,
        lrdmc_alat=-0.20,
        lrdmc_time_branching=0.10,
        lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
        lrdmc_num_walkers=-1,  # default -1 -> num of MPI process.
        lrdmc_twist_average=False,
        lrdmc_kpoints=[],
        lrdmc_force_calc_flag=False,
        lrdmc_maxtime=172000,
    ):
        # job
        self.server_machine_name = server_machine_name
        self.cores = cores
        self.openmp = openmp
        self.queue = queue
        self.version = version
        self.sleep_time = sleep_time
        self.jobpkl_name = jobpkl_name
        # lrdmc
        self.lrdmc_rerun = lrdmc_rerun
        self.lrdmc_max_continuation = lrdmc_max_continuation
        self.lrdmc_pkl_name = lrdmc_pkl_name
        self.lrdmc_target_error_bar = lrdmc_target_error_bar
        self.lrdmc_trial_steps = lrdmc_trial_steps
        self.lrdmc_bin_block = lrdmc_bin_block
        self.lrdmc_warmupblocks = lrdmc_warmupblocks
        self.lrdmc_safe_trial_steps = lrdmc_safe_trial_steps
        self.lrdmc_num_walkers = lrdmc_num_walkers
        self.lrdmc_correcting_factor = lrdmc_correcting_factor
        self.lrdmc_nonlocalmoves = lrdmc_nonlocalmoves
        self.lrdmc_trial_etry = lrdmc_trial_etry
        self.lrdmc_alat = lrdmc_alat
        self.lrdmc_time_branching = lrdmc_time_branching
        self.lrdmc_twist_average = lrdmc_twist_average
        self.lrdmc_kpoints = lrdmc_kpoints
        self.lrdmc_force_calc_flag = lrdmc_force_calc_flag
        self.lrdmc_maxtime = lrdmc_maxtime

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
        self.jobpkl = f"{self.jobpkl_name}.pkl"

        # ******************
        # lrdmc
        # ******************
        os.chdir(self.root_dir)
        self.lrdmc_dir = os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.lrdmc_dir, "pkl")
        logger.info(f"Project root dir = {self.lrdmc_dir}")
        lrdmc_pkl_list = [
            f"{self.lrdmc_pkl_name}_{i}.pkl"
            for i in range(self.lrdmc_max_continuation)
        ]

        #####
        # big loop for all the lrdmc continuations
        #####
        if self.lrdmc_rerun or not all(
            [
                os.path.isfile(os.path.join(self.pkl_dir, lrdmc_pkl))
                for lrdmc_pkl in lrdmc_pkl_list
            ]
        ):
            logger.info("Start: LRDMC calculation")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.lrdmc_dir)

            #####
            # continuation loop !! index is icont
            #####
            for icont in range(self.lrdmc_max_continuation):
                if icont == 0:
                    logger.info(f"LRDMC test run, icont={icont}")
                elif icont == 1:
                    logger.info(f"LRDMC initial run, icont={icont}")
                else:
                    logger.info(f"LRDMC continuation run, icont={icont}")

                self.lrdmc_pkl = f"{self.lrdmc_pkl_name}_{icont}.pkl"
                self.lrdmc_latest_pkl = f"{self.lrdmc_pkl_name}_latest.pkl"
                self.input_file = f"datasfn_{icont}.input"
                self.output_file = f"out_fn_{icont}"

                ####
                # run part
                ####
                if self.lrdmc_rerun or not os.path.isfile(
                    os.path.join(self.lrdmc_dir, self.lrdmc_pkl)
                ):
                    logger.info(
                        f"{self.lrdmc_pkl} does not exist. or lrdmc_rerun = .true."
                    )

                    if icont == 0:
                        self.lrdmc_continuation_flag = False
                        logger.info(
                            f"Run test for estimating steps for achieving the target error bar = {self.lrdmc_target_error_bar}"
                        )

                        if (
                            self.lrdmc_trial_steps
                            <= self.lrdmc_bin_block * self.lrdmc_warmupblocks
                        ):
                            logger.error(
                                f"lrdmcsteps = {self.lrdmc_trial_steps} is smaller than < bin_block * warmupblocks."
                            )
                            raise ValueError

                        # estimated necesary steps per optimization
                        # to achieve the target error bar.
                        if self.lrdmc_safe_trial_steps:
                            lrdmc_minimum_trial_blocks = 40
                            if (
                                self.lrdmc_trial_steps
                                < lrdmc_minimum_trial_blocks
                                * self.lrdmc_bin_block
                                + self.lrdmc_bin_block
                                * self.lrdmc_warmupblocks
                            ):
                                logger.warning(
                                    f"lrdmcsteps = {self.lrdmc_trial_steps} is too small! < {lrdmc_minimum_trial_blocks} * bin_block + bin_block * warmupblocks = {lrdmc_minimum_trial_blocks * self.lrdmc_bin_block + self.lrdmc_bin_block * self.lrdmc_warmupblocks}"
                                )
                                logger.warning(
                                    f"lrdmcsteps = {self.lrdmc_trial_steps} is set to {lrdmc_minimum_trial_blocks} * bin_block + bin_block * warmupblocks = {lrdmc_minimum_trial_blocks * self.lrdmc_bin_block + self.lrdmc_bin_block * self.lrdmc_warmupblocks}"
                                )
                                self.lrdmc_trial_steps = (
                                    lrdmc_minimum_trial_blocks
                                    * self.lrdmc_bin_block
                                    + self.lrdmc_bin_block
                                    * self.lrdmc_warmupblocks
                                )
                        lrdmc_steps = self.lrdmc_trial_steps

                    else:
                        self.lrdmc_continuation_flag = True
                        plrdmc_pkl = f"{self.lrdmc_pkl_name}_{icont-1}.pkl"
                        with open(
                            os.path.join(self.lrdmc_dir, plrdmc_pkl), "rb"
                        ) as f:
                            lrdmc_genius = pickle.load(f)
                        mcmc_steps = get_linenum_fort12(
                            os.path.join(self.lrdmc_dir, "fort.12")
                        )
                        energy, error = (
                            lrdmc_genius.energy,
                            lrdmc_genius.energy_error,
                        )
                        logger.info(
                            f"The errorbar of lrdmc energy {error:.5f} Ha per mcmc step={(mcmc_steps - self.lrdmc_bin_block * self.lrdmc_warmupblocks)}"
                        )
                        if error < self.lrdmc_target_error_bar:
                            logger.warning(
                                f"The target errorbar {self.lrdmc_target_error_bar} Ha has been already achieved!"
                            )
                            logger.warning(
                                "Exiting from the lrdmc continuation loop."
                            )

                            self.output_values["energy"] = energy
                            self.output_values["error"] = error

                            logger.info("LRDMC workflow ends.")
                            os.chdir(self.root_dir)

                            self.status = "success"
                            p_list = [
                                pathlib.Path(ob)
                                for ob in glob.glob(
                                    os.path.join(self.root_dir, "*")
                                )
                            ]
                            self.output_files = [
                                str(p.resolve().relative_to(self.root_dir))
                                for p in p_list
                            ]
                            return (
                                self.status,
                                self.output_files,
                                self.output_values,
                            )

                        lrdmc_steps_estimated_proper = int(
                            (
                                mcmc_steps
                                - self.lrdmc_bin_block
                                * self.lrdmc_warmupblocks
                            )
                            * (error / self.lrdmc_target_error_bar) ** 2
                        )
                        logger.info(
                            f"The target error bar is {self.lrdmc_target_error_bar:.5f} Ha"
                        )
                        logger.info(
                            f"The estimated steps to achieve the target error bar is {lrdmc_steps_estimated_proper:d} steps"
                        )

                        estimated_time_for_1_generation = (
                            lrdmc_genius.estimated_time_for_1_generation
                        )
                        estimated_time = (
                            estimated_time_for_1_generation
                            * lrdmc_steps_estimated_proper
                        )
                        logger.info(
                            f"Estimated time = {estimated_time:.0f} sec."
                        )

                        lrdmc_steps = lrdmc_steps_estimated_proper

                    # generate a lrdmc instance
                    lrdmc_genius = LRDMC_genius(
                        lrdmcsteps=lrdmc_steps,
                        num_walkers=self.lrdmc_num_walkers,
                        alat=self.lrdmc_alat,
                        time_branching=self.lrdmc_time_branching,
                        etry=self.lrdmc_trial_etry,
                        nonlocalmoves=self.lrdmc_nonlocalmoves,
                        twist_average=self.lrdmc_twist_average,
                        kpoints=self.lrdmc_kpoints,
                        force_calc_flag=self.lrdmc_force_calc_flag,
                        maxtime=self.lrdmc_maxtime,
                    )
                    # manual k points!!
                    # if len(self.lrdmc_kpoints) != 0 and self.lrdmc_twist_average == 2:
                    # lrdmc_genius.manual_kpoints=self.lrdmc_kpoints

                    lrdmc_genius.generate_input(
                        input_name=self.input_file,
                        cont=self.lrdmc_continuation_flag,
                    )

                    # binary set
                    if self.cores == self.openmp:
                        binary = "turborvb-serial.x"
                        nompi = True
                    else:
                        binary = "turborvb-mpi.x"
                        nompi = False

                    # Job submission by the job-manager package
                    job = Job_submission(
                        local_machine_name="localhost",
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
                        pkl_name=self.jobpkl,
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
                        os.chdir(self.lrdmc_dir)
                        job_submission_flag, job_number = job.job_submit(
                            submission_script="submit.sh"
                        )
                    logger.info("Job submitted.")

                    with open(
                        os.path.join(self.lrdmc_dir, self.lrdmc_pkl), "wb"
                    ) as f:
                        pickle.dump(lrdmc_genius, f)

                else:
                    logger.info(f"{self.lrdmc_pkl} exists.")
                    with open(self.jobpkl, "rb") as f:
                        job = pickle.load(f)
                    with open(
                        os.path.join(self.lrdmc_dir, self.lrdmc_pkl), "rb"
                    ) as f:
                        lrdmc_genius = pickle.load(f)

                ####
                # Fetch part
                ####
                if self.lrdmc_rerun or not os.path.isfile(
                    os.path.join(self.pkl_dir, self.lrdmc_pkl)
                ):
                    logger.info(
                        f"{self.lrdmc_pkl} does not exist in {self.pkl_dir}."
                    )
                    logger.info(
                        "job is running or fetch has not been done yet."
                    )
                    # job waiting
                    job_running = job.jobcheck()
                    while job_running:
                        logger.info(
                            f"Waiting for the submitted job = {job.job_number}"
                        )
                        # time.sleep(self.sleep_time)
                        await asyncio.sleep(self.sleep_time)
                        os.chdir(self.lrdmc_dir)
                        job_running = job.jobcheck()
                    logger.info("Job finished.")
                    # job fecth
                    logger.info("Fetch files.")
                    fetch_files = [
                        self.output_file,
                        "fort.11",
                        "fort.12",
                        "parminimized.d",
                    ]
                    exclude_files = []
                    if self.lrdmc_twist_average:
                        fetch_files += ["kp_info.dat", "turborvb.scratch"]
                        exclude_files += ["kelcont*", "randseed*"]
                    job.fetch_job(
                        from_objects=fetch_files, exclude_list=exclude_files
                    )
                    logger.info("Fetch finished.")

                    logger.info("Computing lrdmc energy")
                    lrdmc_genius.compute_energy_and_forces(
                        bin_block=self.lrdmc_bin_block,
                        warmupblocks=self.lrdmc_warmupblocks,
                        correcting_factor=self.lrdmc_correcting_factor,
                        rerun=True,
                    )
                    lrdmc_genius.store_result(
                        bin_block=self.lrdmc_bin_block,
                        warmupblocks=self.lrdmc_warmupblocks,
                        output_names=[f"out_fn_{i}" for i in range(icont + 1)],
                    )
                    energy, error = (
                        lrdmc_genius.energy,
                        lrdmc_genius.energy_error,
                    )
                    estimated_time_for_1_generation = (
                        lrdmc_genius.estimated_time_for_1_generation
                    )
                    logger.info(
                        f"LRDMC energy = {energy:.5f} +- {error:3f} Ha"
                    )
                    logger.info(
                        f"estimated_time_for_1_generation = {estimated_time_for_1_generation:.5f} sec"
                    )
                    self.output_values["energy"] = energy
                    self.output_values["error"] = error

                    with open(
                        os.path.join(self.lrdmc_dir, self.lrdmc_pkl), "wb"
                    ) as f:
                        pickle.dump(lrdmc_genius, f)
                    with open(
                        os.path.join(self.pkl_dir, self.lrdmc_pkl), "wb"
                    ) as f:
                        pickle.dump(lrdmc_genius, f)
                    with open(
                        os.path.join(self.pkl_dir, self.lrdmc_latest_pkl), "wb"
                    ) as f:
                        pickle.dump(lrdmc_genius, f)

                logger.info(f"LRDMC run ends for icont={icont}")

            os.chdir(self.root_dir)

        else:
            logger.info("Skip: LRDMC calculation")
            self.lrdmc_latest_pkl = f"{self.lrdmc_pkl_name}_latest.pkl"
            with open(
                os.path.join(self.pkl_dir, self.lrdmc_latest_pkl), "rb"
            ) as f:
                lrdmc_genius = pickle.load(f)
            energy, error = lrdmc_genius.energy, lrdmc_genius.energy_error
            logger.info(f"LRDMC energy = {energy:.5f} +- {error:3f} Ha")
            self.output_values["energy"] = energy
            self.output_values["error"] = error

        logger.info("LRDMC workflow ends.")
        os.chdir(self.root_dir)

        self.status = "success"
        p_list = [
            pathlib.Path(ob)
            for ob in glob.glob(os.path.join(self.root_dir, "*"))
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
    handler_format = Formatter(
        "%(name)s - %(levelname)s - %(lineno)d - %(message)s"
    )
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    # moved to examples
