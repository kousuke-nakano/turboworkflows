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

# turbo-genius packages
from turbogenius.lrdmc_opt_genius import LRDMCopt_genius
from turbogenius.pyturbo.lrdmcopt import LRDMCopt

# jobmanager
from turbofilemanager.job_manager import Job_submission

# turboworkflow packages
from turboworkflows.workflow_encapsulated import Workflow

logger = getLogger("Turbo-Workflows").getChild(__name__)


class LRDMCopt_workflow(Workflow):
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
        # lrdmcopt
        lrdmcopt_max_continuation=2,
        lrdmcopt_pkl_name="lrdmcopt_genius",
        lrdmcopt_target_error_bar=1.0e-3,  # Ha
        lrdmcopt_trial_optsteps=50,
        lrdmcopt_trial_steps=50,
        lrdmcopt_minimum_blocks=3,
        lrdmcopt_production_optsteps=2000,
        lrdmcopt_optwarmupsteps_ratio=0.8,
        lrdmcopt_bin_block=1,
        lrdmcopt_warmupblocks=0,
        lrdmcopt_optimizer="sr",
        lrdmcopt_learning_rate=0.002,
        lrdmcopt_regularization=0.001,
        lrdmcopt_alat=-0.20,
        lrdmcopt_trial_etry=0.0,
        lrdmcopt_nonlocalmoves="dlatm",  # tmove, dla, dlatm
        lrdmcopt_onebody=False,
        lrdmcopt_twobody=False,
        lrdmcopt_det_mat=True,
        lrdmcopt_jas_mat=False,
        lrdmcopt_det_basis_exp=False,
        lrdmcopt_jas_basis_exp=False,
        lrdmcopt_det_basis_coeff=False,
        lrdmcopt_jas_basis_coeff=False,
        lrdmcopt_num_walkers=-1,  # default -1 -> num of MPI process.
        lrdmcopt_twist_average=False,
        lrdmcopt_kpoints=[],
        lrdmcopt_maxtime=172000,
    ):
        # job
        self.server_machine_name = server_machine_name
        self.cores = cores
        self.openmp = openmp
        self.queue = queue
        self.version = version
        self.sleep_time = sleep_time
        self.jobpkl_name = jobpkl_name
        # lrdmcopt
        self.lrdmcopt_rerun = False
        self.lrdmcopt_max_continuation = lrdmcopt_max_continuation
        self.lrdmcopt_pkl_name = lrdmcopt_pkl_name
        self.lrdmcopt_target_error_bar = lrdmcopt_target_error_bar
        self.lrdmcopt_trial_optsteps = lrdmcopt_trial_optsteps
        self.lrdmcopt_trial_steps = lrdmcopt_trial_steps
        self.lrdmcopt_minimum_blocks = lrdmcopt_minimum_blocks
        self.lrdmcopt_production_optsteps = lrdmcopt_production_optsteps
        self.lrdmcopt_optwarmupsteps_ratio = lrdmcopt_optwarmupsteps_ratio
        self.lrdmcopt_bin_block = lrdmcopt_bin_block
        self.lrdmcopt_warmupblocks = lrdmcopt_warmupblocks
        self.lrdmcopt_optimizer = lrdmcopt_optimizer
        self.lrdmcopt_learning_rate = lrdmcopt_learning_rate
        self.lrdmcopt_regularization = lrdmcopt_regularization
        self.lrdmcopt_alat = lrdmcopt_alat
        self.lrdmcopt_trial_etry = lrdmcopt_trial_etry
        self.lrdmcopt_nonlocalmoves = lrdmcopt_nonlocalmoves
        self.lrdmcopt_onebody = lrdmcopt_onebody
        self.lrdmcopt_twobody = lrdmcopt_twobody
        self.lrdmcopt_det_mat = lrdmcopt_det_mat
        self.lrdmcopt_jas_mat = lrdmcopt_jas_mat
        self.lrdmcopt_det_basis_exp = lrdmcopt_det_basis_exp
        self.lrdmcopt_jas_basis_exp = lrdmcopt_jas_basis_exp
        self.lrdmcopt_det_basis_coeff = lrdmcopt_det_basis_coeff
        self.lrdmcopt_jas_basis_coeff = lrdmcopt_jas_basis_coeff
        self.lrdmcopt_num_walkers = lrdmcopt_num_walkers
        self.lrdmcopt_twist_average = lrdmcopt_twist_average
        self.lrdmcopt_kpoints = lrdmcopt_kpoints
        self.lrdmcopt_maxtime = lrdmcopt_maxtime
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
        # LRDMCopt
        # ******************
        os.chdir(self.root_dir)
        self.lrdmcopt_dir = os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.lrdmcopt_dir, "pkl")
        logger.info(f"Project root dir = {self.lrdmcopt_dir}")
        lrdmcopt_pkl_list = [
            f"{self.lrdmcopt_pkl_name}_{i}.pkl"
            for i in range(self.lrdmcopt_max_continuation)
        ]

        #####
        # big loop for all the LRDMCopt continuations
        #####
        if self.lrdmcopt_rerun or not all(
            [
                os.path.isfile(os.path.join(self.pkl_dir, lrdmcopt_pkl))
                for lrdmcopt_pkl in lrdmcopt_pkl_list
            ]
        ):
            logger.info("Start: LRDMCopt calculation")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.lrdmcopt_dir)

            #####
            # continuation loop !! index is icont
            #####
            for icont in range(self.lrdmcopt_max_continuation):
                if icont == 0:
                    logger.info(f"LRDMCopt test run, icont={icont}")
                elif icont == 1:
                    logger.info(f"LRDMCopt initial run, icont={icont}")
                else:
                    logger.info(f"LRDMCopt continuation run, icont={icont}")

                self.lrdmcopt_pkl = f"{self.lrdmcopt_pkl_name}_{icont}.pkl"
                self.lrdmcopt_latest_pkl = (
                    f"{self.lrdmcopt_pkl_name}_latest.pkl"
                )
                self.input_file = f"datasfn_opt_{icont}.input"
                self.output_file = f"out_fn_opt_{icont}"

                ####
                # run part
                ####
                if self.lrdmcopt_rerun or not os.path.isfile(
                    os.path.join(self.lrdmcopt_dir, self.lrdmcopt_pkl)
                ):
                    logger.info(
                        f"{self.lrdmcopt_pkl} does not exist. or lrdmcopt_rerun = .true."
                    )

                    if icont == 0:
                        self.lrdmcopt_continuation_flag = False
                        logger.info(
                            f"Run test for estimating steps for achieving the target error bar = {self.lrdmcopt_target_error_bar}"
                        )
                        if (
                            self.lrdmcopt_trial_steps
                            <= self.lrdmcopt_bin_block
                            * self.lrdmcopt_warmupblocks
                        ):
                            logger.error(
                                "lrdmcopt_trial_steps <= lrdmcopt_bin_block * lrdmcopt_warmupblocks"
                            )
                            raise ValueError
                        lrdmcoptsteps = self.lrdmcopt_trial_optsteps
                        steps = self.lrdmcopt_trial_steps

                    else:
                        self.lrdmcopt_continuation_flag = True
                        pinput_file = f"datasfn_opt_{icont-1}.input"
                        plrdmcopt_pkl = (
                            f"{self.lrdmcopt_pkl_name}_{icont-1}.pkl"
                        )
                        with open(
                            os.path.join(self.lrdmcopt_dir, plrdmcopt_pkl),
                            "rb",
                        ) as f:
                            lrdmcopt_genius = pickle.load(f)
                        logger.info("LRDMC optimization, production run step")
                        energy, error = (
                            lrdmcopt_genius.energy,
                            lrdmcopt_genius.energy_error,
                        )
                        logger.info(
                            f"The lrdmc energy at the final step in the test run is {energy[-1]:.5f} Ha"
                        )
                        lrdmcopt_pyturbo = LRDMCopt.parse_from_file(
                            file=pinput_file,
                            in_fort10="fort.10",
                            twist_average=self.lrdmcopt_twist_average,
                        )
                        nweight = lrdmcopt_pyturbo.get_parameter(
                            parameter="nweight"
                        )
                        logger.info(
                            f"The error bar of the lrdmc energy at the final step is {error[-1]:.5f} Ha per mcmc step={(nweight - self.lrdmcopt_warmupblocks * self.lrdmcopt_bin_block)}"
                        )
                        lrdmcopt_steps_estimated_proper = int(
                            (
                                nweight
                                - self.lrdmcopt_warmupblocks
                                * self.lrdmcopt_bin_block
                            )
                            * (error[-1] / self.lrdmcopt_target_error_bar) ** 2
                        )
                        logger.info(
                            f"The target error bar per optstep is {self.lrdmcopt_target_error_bar:.5f} Ha"
                        )
                        logger.info(
                            f"The estimated steps to achieve the target error bar is {lrdmcopt_steps_estimated_proper:d} steps"
                        )
                        if (
                            lrdmcopt_steps_estimated_proper
                            < (
                                self.lrdmcopt_warmupblocks
                                + self.lrdmcopt_minimum_blocks
                            )
                            * self.lrdmcopt_bin_block
                        ):
                            lrdmcopt_steps_estimated_proper = (
                                self.lrdmcopt_warmupblocks
                                + self.lrdmcopt_minimum_blocks
                            ) * self.lrdmcopt_bin_block
                            logger.warning(
                                f"lrdmcopt_steps_estimated_proper is set to {lrdmcopt_steps_estimated_proper}"
                            )
                        estimated_time_for_1_generation = (
                            lrdmcopt_genius.estimated_time_for_1_generation
                        )
                        estimated_time = (
                            estimated_time_for_1_generation
                            * lrdmcopt_steps_estimated_proper
                            * self.lrdmcopt_production_optsteps
                        )
                        logger.info(
                            f"Estimated time = {estimated_time:.0f} sec."
                        )

                        lrdmcoptsteps = self.lrdmcopt_production_optsteps
                        steps = lrdmcopt_steps_estimated_proper

                    # generate a LRDMCopt instance
                    lrdmcopt_genius = LRDMCopt_genius(
                        lrdmcoptsteps=lrdmcoptsteps,
                        steps=steps,
                        bin_block=self.lrdmcopt_bin_block,
                        warmupblocks=self.lrdmcopt_warmupblocks,
                        num_walkers=self.lrdmcopt_num_walkers,
                        optimizer=self.lrdmcopt_optimizer,
                        learning_rate=self.lrdmcopt_learning_rate,
                        regularization=self.lrdmcopt_regularization,
                        alat=self.lrdmcopt_alat,
                        etry=self.lrdmcopt_trial_etry,
                        nonlocalmoves=self.lrdmcopt_nonlocalmoves,
                        opt_onebody=self.lrdmcopt_onebody,
                        opt_twobody=self.lrdmcopt_twobody,
                        opt_det_mat=self.lrdmcopt_det_mat,
                        opt_jas_mat=self.lrdmcopt_jas_mat,
                        opt_det_basis_exp=self.lrdmcopt_det_basis_exp,
                        opt_jas_basis_exp=self.lrdmcopt_jas_basis_exp,
                        opt_det_basis_coeff=self.lrdmcopt_det_basis_coeff,
                        opt_jas_basis_coeff=self.lrdmcopt_jas_basis_coeff,
                        twist_average=self.lrdmcopt_twist_average,
                        kpoints=self.lrdmcopt_kpoints,
                        maxtime=self.lrdmcopt_maxtime,
                    )
                    # manual k points!!
                    # if len(self.lrdmcopt_kpoints) != 0 and self.lrdmcopt_twist_average == 2:
                    # lrdmcopt_genius.manual_kpoints=self.lrdmcopt_kpoints

                    lrdmcopt_genius.generate_input(
                        input_name=self.input_file,
                        cont=self.lrdmcopt_continuation_flag,
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
                        os.chdir(self.lrdmcopt_dir)
                        job_submission_flag, job_number = job.job_submit(
                            submission_script="submit.sh"
                        )
                    logger.info("Job submitted.")

                    with open(
                        os.path.join(self.lrdmcopt_dir, self.lrdmcopt_pkl),
                        "wb",
                    ) as f:
                        pickle.dump(lrdmcopt_genius, f)

                else:
                    logger.info(f"{self.lrdmcopt_pkl} exists.")
                    with open(self.jobpkl, "rb") as f:
                        job = pickle.load(f)
                    with open(
                        os.path.join(self.lrdmcopt_dir, self.lrdmcopt_pkl),
                        "rb",
                    ) as f:
                        lrdmcopt_genius = pickle.load(f)

                ####
                # Fetch part
                ####
                if self.lrdmcopt_rerun or not os.path.isfile(
                    os.path.join(self.pkl_dir, self.lrdmcopt_pkl)
                ):
                    logger.info(
                        f"{self.lrdmcopt_pkl} does not exist in {self.pkl_dir}."
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
                        os.chdir(self.lrdmcopt_dir)
                        job_running = job.jobcheck()
                    logger.info("Job finished.")
                    # job fecth
                    logger.info("Fetch files.")
                    fetch_files = [
                        self.output_file,
                        "fort.10",
                        "fort.11",
                        "fort.12",
                        "parminimized.d",
                        "forces.dat",
                    ]
                    exclude_files = []
                    if self.lrdmcopt_twist_average:
                        fetch_files += ["kp_info.dat", "turborvb.scratch"]
                        exclude_files += ["kelcont*", "randseed*"]
                    job.fetch_job(
                        from_objects=fetch_files, exclude_list=exclude_files
                    )
                    logger.info("Fetch finished.")

                    lrdmcopt_genius.store_result(
                        output_names=[self.output_file]
                    )
                    lrdmcopt_genius.plot_energy_and_devmax(
                        output_names=[
                            f"out_fn_opt_{i}" for i in range(icont + 1)
                        ],
                        interactive=False,
                    )
                    if icont > 0:
                        with open("forces.dat", "r") as f:
                            lines = f.readlines()
                        lrdmcopt_done_optsteps = len(lines)
                        optwarmupsteps = int(
                            self.lrdmcopt_optwarmupsteps_ratio
                            * lrdmcopt_done_optsteps
                        )
                        logger.info(
                            f"optwarmupsteps is set to {optwarmupsteps} (the first {self.lrdmcopt_optwarmupsteps_ratio*100:.0f}% steps are disregarded.)"
                        )
                        logger.info(
                            f"The final {(1-self.lrdmcopt_optwarmupsteps_ratio)*100:.0f}% steps will be used for averaging parameters."
                        )
                        lrdmcopt_genius.average(
                            optwarmupsteps=optwarmupsteps,
                            input_name=self.input_file,
                            output_names=[
                                f"out_fn_opt_{i}" for i in range(icont + 1)
                            ],
                            graph_plot=True,
                        )

                    with open(
                        os.path.join(self.lrdmcopt_dir, self.lrdmcopt_pkl),
                        "wb",
                    ) as f:
                        pickle.dump(lrdmcopt_genius, f)
                    with open(
                        os.path.join(self.pkl_dir, self.lrdmcopt_pkl), "wb"
                    ) as f:
                        pickle.dump(lrdmcopt_genius, f)
                    with open(
                        os.path.join(self.pkl_dir, self.lrdmcopt_latest_pkl),
                        "wb",
                    ) as f:
                        pickle.dump(lrdmcopt_genius, f)

                logger.info(f"LRDMCopt run ends for icont={icont}")

            os.chdir(self.root_dir)

        else:
            logger.info("Skip: LRDMCopt calculation")
            self.lrdmcopt_latest_pkl = f"{self.lrdmcopt_pkl_name}_latest.pkl"
            with open(
                os.path.join(self.pkl_dir, self.lrdmcopt_latest_pkl), "rb"
            ) as f:
                lrdmcopt_genius = pickle.load(f)

        logger.info("LRDMCopt workflow ends.")
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
