#!/usr/bin/env python
# coding: utf-8

# python packages
import os
import pickle
import asyncio
import glob
import pathlib
from typing import Optional

# Logger
from logging import getLogger, StreamHandler, Formatter

# turbo-genius packages
from turbogenius.vmc_opt_genius import VMCopt_genius
from turbogenius.pyturbo.vmcopt import VMCopt

# jobmanager
from turbofilemanager.job_manager import Job_submission

# turboworkflow packages
from turboworkflows.workflow_encapsulated import Workflow

logger = getLogger("Turbo-Workflows").getChild(__name__)


class VMCopt_workflow(Workflow):
    def __init__(
        self,
        # job
        server_machine_name: str = "localhost",
        cores: int = 1,
        openmp: int = 1,
        queue: Optional[str] = None,
        version: str = "stable",
        sleep_time: int = 1800,  # sec.
        jobpkl_name: str = "job_manager",
        # vmcopt
        vmcopt_max_continuation: int = 2,
        vmcopt_pkl_name: str = "vmcopt_genius",
        vmcopt_target_error_bar: float = 1.0e-3,  # Ha
        vmcopt_trial_optsteps: float = 50,
        vmcopt_trial_steps: float = 50,
        vmcopt_minimum_blocks: int = 3,
        vmcopt_production_optsteps: int = 2000,
        vmcopt_optwarmupsteps_ratio: int = 0.8,
        vmcopt_bin_block: int = 1,
        vmcopt_warmupblocks: int = 0,
        vmcopt_optimizer: str = "lr",
        vmcopt_learning_rate: float = 0.35,
        vmcopt_regularization: float = 0.001,
        vmcopt_onebody: bool = True,
        vmcopt_twobody: bool = True,
        vmcopt_det_mat: bool = False,
        vmcopt_jas_mat: bool = True,
        vmcopt_det_basis_exp: bool = False,
        vmcopt_jas_basis_exp: bool = False,
        vmcopt_det_basis_coeff: bool = False,
        vmcopt_jas_basis_coeff: bool = False,
        vmcopt_num_walkers: int = -1,  # default -1 -> num of MPI process.
        vmcopt_twist_average: bool = False,
        vmcopt_kpoints: Optional[list] = None,
        vmcopt_maxtime: int = 172000,
    ):
        if vmcopt_kpoints is None:
            vmcopt_kpoints = []
        # job
        self.server_machine_name = server_machine_name
        self.cores = cores
        self.openmp = openmp
        self.queue = queue
        self.version = version
        self.sleep_time = sleep_time
        self.jobpkl_name = jobpkl_name
        # vmcopt
        self.vmcopt_rerun = False
        self.vmcopt_max_continuation = vmcopt_max_continuation
        self.vmcopt_pkl_name = vmcopt_pkl_name
        self.vmcopt_target_error_bar = vmcopt_target_error_bar
        self.vmcopt_trial_optsteps = vmcopt_trial_optsteps
        self.vmcopt_trial_steps = vmcopt_trial_steps
        self.vmcopt_minimum_blocks = vmcopt_minimum_blocks
        self.vmcopt_production_optsteps = vmcopt_production_optsteps
        self.vmcopt_optwarmupsteps_ratio = vmcopt_optwarmupsteps_ratio
        self.vmcopt_bin_block = vmcopt_bin_block
        self.vmcopt_warmupblocks = vmcopt_warmupblocks
        self.vmcopt_optimizer = vmcopt_optimizer
        self.vmcopt_learning_rate = vmcopt_learning_rate
        self.vmcopt_regularization = vmcopt_regularization
        self.vmcopt_onebody = vmcopt_onebody
        self.vmcopt_twobody = vmcopt_twobody
        self.vmcopt_det_mat = vmcopt_det_mat
        self.vmcopt_jas_mat = vmcopt_jas_mat
        self.vmcopt_det_basis_exp = vmcopt_det_basis_exp
        self.vmcopt_jas_basis_exp = vmcopt_jas_basis_exp
        self.vmcopt_det_basis_coeff = vmcopt_det_basis_coeff
        self.vmcopt_jas_basis_coeff = vmcopt_jas_basis_coeff
        self.vmcopt_num_walkers = vmcopt_num_walkers
        self.vmcopt_twist_average = vmcopt_twist_average
        self.vmcopt_kpoints = vmcopt_kpoints
        self.vmcopt_maxtime = vmcopt_maxtime
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
        # VMCopt
        # ******************
        os.chdir(self.root_dir)
        self.vmcopt_dir = os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.vmcopt_dir, "pkl")
        logger.info(f"Project root dir = {self.vmcopt_dir}")
        vmcopt_pkl_list = [
            f"{self.vmcopt_pkl_name}_{i}.pkl"
            for i in range(self.vmcopt_max_continuation)
        ]

        #####
        # big loop for all the VMCopt continuations
        #####
        if self.vmcopt_rerun or not all(
            [
                os.path.isfile(os.path.join(self.pkl_dir, vmcopt_pkl))
                for vmcopt_pkl in vmcopt_pkl_list
            ]
        ):
            logger.info("Start: VMCopt calculation")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.vmcopt_dir)

            #####
            # continuation loop !! index is icont
            #####
            for icont in range(self.vmcopt_max_continuation):
                if icont == 0:
                    logger.info(f"VMCopt test run, icont={icont}")
                elif icont == 1:
                    logger.info(f"VMCopt initial run, icont={icont}")
                else:
                    logger.info(f"VMCopt continuation run, icont={icont}")

                self.vmcopt_pkl = f"{self.vmcopt_pkl_name}_{icont}.pkl"
                self.vmcopt_latest_pkl = f"{self.vmcopt_pkl_name}_latest.pkl"
                self.input_file = f"datasmin_{icont}.input"
                self.output_file = f"out_min_{icont}"

                ####
                # run part
                ####
                if self.vmcopt_rerun or not os.path.isfile(
                    os.path.join(self.vmcopt_dir, self.vmcopt_pkl)
                ):
                    logger.info(
                        f"{self.vmcopt_pkl} does not exist. or vmcopt_rerun = .true."
                    )

                    if icont == 0:
                        self.vmcopt_continuation_flag = False
                        logger.info(
                            f"Run test for estimating steps for achieving the target error bar = {self.vmcopt_target_error_bar}"
                        )
                        if (
                            self.vmcopt_trial_steps
                            <= self.vmcopt_bin_block * self.vmcopt_warmupblocks
                        ):
                            logger.error(
                                "vmcopt_trial_steps <= vmcopt_bin_block * vmcopt_warmupblocks"
                            )
                            raise ValueError
                        vmcoptsteps = self.vmcopt_trial_optsteps
                        steps = self.vmcopt_trial_steps

                    else:
                        self.vmcopt_continuation_flag = True
                        pinput_file = f"datasmin_{icont-1}.input"
                        pvmcopt_pkl = f"{self.vmcopt_pkl_name}_{icont-1}.pkl"
                        with open(
                            os.path.join(self.vmcopt_dir, pvmcopt_pkl), "rb"
                        ) as f:
                            vmcopt_genius = pickle.load(f)
                        logger.info("VMC optimization, production run step")
                        energy, error = (
                            vmcopt_genius.energy,
                            vmcopt_genius.energy_error,
                        )
                        logger.info(
                            f"The vmc energy at the final step in the test run is {energy[-1]:.5f} Ha"
                        )
                        vmcopt_pyturbo = VMCopt.parse_from_file(
                            file=pinput_file,
                            in_fort10="fort.10",
                            twist_average=self.vmcopt_twist_average,
                        )
                        nweight = vmcopt_pyturbo.get_parameter(
                            parameter="nweight"
                        )
                        logger.info(
                            f"The error bar of the vmc energy at the final step is {error[-1]:.5f} Ha per mcmc step={(nweight - self.vmcopt_warmupblocks * self.vmcopt_bin_block)}"
                        )
                        vmcopt_steps_estimated_proper = int(
                            (
                                nweight
                                - self.vmcopt_warmupblocks
                                * self.vmcopt_bin_block
                            )
                            * (error[-1] / self.vmcopt_target_error_bar) ** 2
                        )
                        logger.info(
                            f"The target error bar per optstep is {self.vmcopt_target_error_bar:.5f} Ha"
                        )
                        logger.info(
                            f"The estimated steps to achieve the target error bar is {vmcopt_steps_estimated_proper:d} steps"
                        )
                        if (
                            vmcopt_steps_estimated_proper
                            < (
                                self.vmcopt_minimum_blocks
                                + self.vmcopt_warmupblocks
                            )
                            * self.vmcopt_bin_block
                        ):
                            vmcopt_steps_estimated_proper = (
                                self.vmcopt_minimum_blocks
                                + self.vmcopt_warmupblocks
                            ) * self.vmcopt_bin_block
                            logger.warning(
                                f"vmcopt_steps_estimated_proper is set to {vmcopt_steps_estimated_proper}"
                            )
                        estimated_time_for_1_generation = (
                            vmcopt_genius.estimated_time_for_1_generation
                        )
                        estimated_time = (
                            estimated_time_for_1_generation
                            * vmcopt_steps_estimated_proper
                            * self.vmcopt_production_optsteps
                        )
                        logger.info(
                            f"Estimated time = {estimated_time:.0f} sec."
                        )

                        vmcoptsteps = self.vmcopt_production_optsteps
                        steps = vmcopt_steps_estimated_proper

                    # generate a VMCopt instance
                    vmcopt_genius = VMCopt_genius(
                        vmcoptsteps=vmcoptsteps,
                        steps=steps,
                        bin_block=self.vmcopt_bin_block,
                        warmupblocks=self.vmcopt_warmupblocks,
                        num_walkers=self.vmcopt_num_walkers,
                        optimizer=self.vmcopt_optimizer,
                        learning_rate=self.vmcopt_learning_rate,
                        regularization=self.vmcopt_regularization,
                        opt_onebody=self.vmcopt_onebody,
                        opt_twobody=self.vmcopt_twobody,
                        opt_det_mat=self.vmcopt_det_mat,
                        opt_jas_mat=self.vmcopt_jas_mat,
                        opt_det_basis_exp=self.vmcopt_det_basis_exp,
                        opt_jas_basis_exp=self.vmcopt_jas_basis_exp,
                        opt_det_basis_coeff=self.vmcopt_det_basis_coeff,
                        opt_jas_basis_coeff=self.vmcopt_jas_basis_coeff,
                        twist_average=self.vmcopt_twist_average,
                        kpoints=self.vmcopt_kpoints,
                        maxtime=self.vmcopt_maxtime,
                    )
                    # manual k points!!
                    # if len(self.vmcopt_kpoints) != 0 and self.vmcopt_twist_average == 2: vmcopt_genius.manual_kpoints=self.vmcopt_kpoints

                    vmcopt_genius.generate_input(
                        input_name=self.input_file,
                        cont=self.vmcopt_continuation_flag,
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
                        os.chdir(self.vmcopt_dir)
                        job_submission_flag, job_number = job.job_submit(
                            submission_script="submit.sh"
                        )
                    logger.info("Job submitted.")

                    with open(
                        os.path.join(self.vmcopt_dir, self.vmcopt_pkl), "wb"
                    ) as f:
                        pickle.dump(vmcopt_genius, f)

                else:
                    logger.info(f"{self.vmcopt_pkl} exists.")
                    with open(self.jobpkl, "rb") as f:
                        job = pickle.load(f)
                    with open(
                        os.path.join(self.vmcopt_dir, self.vmcopt_pkl), "rb"
                    ) as f:
                        vmcopt_genius = pickle.load(f)

                ####
                # Fetch part
                ####
                if self.vmcopt_rerun or not os.path.isfile(
                    os.path.join(self.pkl_dir, self.vmcopt_pkl)
                ):
                    logger.info(
                        f"{self.vmcopt_pkl} does not exist in {self.pkl_dir}."
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
                        os.chdir(self.vmcopt_dir)
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
                    if self.vmcopt_twist_average:
                        fetch_files += ["kp_info.dat", "turborvb.scratch"]
                        exclude_files += ["kelcont*", "randseed*"]
                    job.fetch_job(
                        from_objects=fetch_files, exclude_list=exclude_files
                    )
                    logger.info("Fetch finished.")

                    vmcopt_genius.store_result(output_names=[self.output_file])
                    vmcopt_genius.plot_energy_and_devmax(
                        output_names=[
                            f"out_min_{i}" for i in range(icont + 1)
                        ],
                        interactive=False,
                    )
                    if icont > 0:
                        with open("forces.dat", "r") as f:
                            lines = f.readlines()
                        vmcopt_done_optsteps = len(lines)
                        optwarmupsteps = int(
                            self.vmcopt_optwarmupsteps_ratio
                            * vmcopt_done_optsteps
                        )
                        logger.info(
                            f"optwarmupsteps is set to {optwarmupsteps} (the first {self.vmcopt_optwarmupsteps_ratio*100:.0f}% steps are disregarded.)"
                        )
                        logger.info(
                            f"The final {(1-self.vmcopt_optwarmupsteps_ratio)*100:.0f}% steps will be used for averaging parameters."
                        )
                        vmcopt_genius.average(
                            optwarmupsteps=optwarmupsteps,
                            input_name=self.input_file,
                            output_names=[
                                f"out_min_{i}" for i in range(icont + 1)
                            ],
                            graph_plot=True,
                        )

                    with open(
                        os.path.join(self.vmcopt_dir, self.vmcopt_pkl), "wb"
                    ) as f:
                        pickle.dump(vmcopt_genius, f)
                    with open(
                        os.path.join(self.pkl_dir, self.vmcopt_pkl), "wb"
                    ) as f:
                        pickle.dump(vmcopt_genius, f)
                    with open(
                        os.path.join(self.pkl_dir, self.vmcopt_latest_pkl),
                        "wb",
                    ) as f:
                        pickle.dump(vmcopt_genius, f)

                logger.info(f"VMCopt run ends for icont={icont}")

            os.chdir(self.root_dir)

        else:
            logger.info("Skip: VMCopt calculation")
            self.vmcopt_latest_pkl = f"{self.vmcopt_pkl_name}_latest.pkl"
            with open(
                os.path.join(self.pkl_dir, self.vmcopt_latest_pkl), "rb"
            ) as f:
                vmcopt_genius = pickle.load(f)

        logger.info("VMCopt workflow ends.")
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
