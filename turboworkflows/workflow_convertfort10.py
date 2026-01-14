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

# turboworkflows packages
from .turbofilemanager.job_manager import Job_submission
from .workflow_encapsulated import Workflow

# turbo-genius packages
from turbogenius.convertfort10_genius import Convertfort10_genius

logger = getLogger("Turbo-Workflows").getChild(__name__)


# convertfort10
class Convertfort10_workflow(Workflow):
    def __init__(
        self,
        # job
        server_machine_name: str = "localhost",
        queue_label: Optional[str] = "default",
        mpi: bool = False,
        version: str = "stable",
        sleep_time: int = 1800,  # sec.
        # convertfort10
        convertfort10_rerun: bool = False,
        in_fort10: str = "fort.10_in",
        out_fort10: str = "fort.10_out",
        grid_size: float = 0.10,
    ):
        # job
        self.server_machine_name = server_machine_name
        self.mpi = mpi
        self.queue_label = queue_label
        self.version = version
        self.sleep_time = sleep_time
        # convertfort10
        self.convertfort10_rerun = convertfort10_rerun
        self.in_fort10 = in_fort10
        self.out_fort10 = out_fort10
        self.grid_size = grid_size
        # pkl names
        self.job_pkl_name = "job_manager"
        self.convertfort10mol_pkl_name = "convertfort10_genius"
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
        # convertfort10
        # ******************
        os.chdir(self.root_dir)
        self.convertfort10_dir = os.path.join(self.root_dir)
        self.pkl_dir = os.path.join(self.convertfort10_dir, "pkl")
        logger.info(f"Project root dir = {self.convertfort10_dir}")
        self.convertfort10_pkl = f"{self.convertfort10_pkl_name}.pkl"

        if self.convertfort10_rerun or not os.path.isfile(
            os.path.join(self.pkl_dir, self.convertfort10_pkl)
        ):
            logger.info("Start: convertfort10")
            os.makedirs(self.pkl_dir, exist_ok=True)
            os.chdir(self.convertfort10_dir)
            self.input_file = "convertfort10.input"
            self.output_file = "out_mol"

            if self.convertfort10_rerun or not os.path.isfile(
                os.path.join(self.convertfort10_dir, self.convertfort10_pkl)
            ):

                convertfort10_genius = Convertfort10_genius(
                    in_fort10=self.in_fort10,
                    out_fort10=self.out_fort10,
                    grid_size=self.grid_size,
                )

                convertfort10_genius.generate_input(input_name=self.input_file)

                # binary set
                if self.mpi:
                    binary = "convertfort10-mpi.x"
                else:
                    binary = "convertfort10.x"

                # Job submission by the job-manager package
                job = Job_submission(
                    client_machine_name="localhost",
                    server_machine_name=self.server_machine_name,
                    package="turborvb",
                    queue_label=self.queue_label,
                    version=self.version,
                    binary=binary,
                    mpi=self.mpi,
                    jobname="turbogenius",
                    input_file=self.input_file,
                    output_file=self.output_file,
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
                    os.chdir(self.convertfort10_dir)
                    job_submission_flag, job_number = job.job_submit(
                        submission_script="submit.sh"
                    )
                logger.info("Job submitted.")

                with open(
                    os.path.join(self.convertfort10_dir, self.convertfort10_pkl),
                    "wb",
                ) as f:
                    pickle.dump(convertfort10_genius, f)

            else:
                logger.info(f"{self.convertfort10_pkl} exists.")
                with open(self.job_pkl, "rb") as f:
                    job = pickle.load(f)
                with open(
                    os.path.join(self.convertfort10_dir, self.convertfort10_pkl),
                    "rb",
                ) as f:
                    convertfort10_genius = pickle.load(f)

            ####
            # Fetch part
            ####
            if self.convertfort10_rerun or not os.path.isfile(
                os.path.join(self.pkl_dir, self.convertfort10_pkl)
            ):
                logger.info(
                    f"{self.convertfort10_pkl} does not exist in {self.pkl_dir}."
                )
                logger.info("job is running or fetch has not been done yet.")
                # job waiting
                job_running = job.jobcheck()
                while job_running:
                    logger.info(f"Waiting for the submitted job = {job.job_number}")
                    # time.sleep(self.sleep_time)
                    await asyncio.sleep(self.sleep_time)
                    os.chdir(self.convertfort10_dir)
                    job_running = job.jobcheck()
                logger.info("Job finished.")
                # job fecth
                logger.info("Fetch files.")
                fetch_files = [self.output_file, "fort.10_new"]
                exclude_patterns = []
                job.fetch_job(
                    from_objects=fetch_files, exclude_patterns=exclude_patterns
                )
                logger.info("Fetch finished.")

                with open(
                    os.path.join(self.convertfort10_dir, self.convertfort10_pkl),
                    "wb",
                ) as f:
                    pickle.dump(convertfort10_genius, f)
                with open(
                    os.path.join(self.pkl_dir, self.convertfort10_pkl), "wb"
                ) as f:
                    pickle.dump(convertfort10_genius, f)

                os.chdir(self.root_dir)

        else:
            logger.info("Skip: convertfort10")
            self.convertfort10_pkl = f"{self.convertfort10_pkl_name}.pkl"
            with open(os.path.join(self.pkl_dir, self.convertfort10_pkl), "rb") as f:
                convertfort10_genius = pickle.load(f)

        logger.info("End: convertfort10 workflow ends.")
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

    # moved to examples
