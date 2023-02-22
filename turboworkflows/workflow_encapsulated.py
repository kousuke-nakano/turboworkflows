#!/usr/bin/env python
# coding: utf-8

# python packages
import os
import shutil
import asyncio
from typing import Optional

# Logger
from logging import getLogger, StreamHandler, Formatter

logger = getLogger("Turbo-Workflows").getChild(__name__)


class Workflow:
    def __init__(self):
        # return values
        self.status = "init"
        self.output_files = []
        self.output_values = {}

    async def async_launch(
        self,
    ):  # --> return self.status, self.output_files, self.output_values
        return self.status, self.output_files, self.output_values

    def launch(self):
        return asyncio.run(self.async_launch())


class Encapsulated_Workflow:
    def __init__(
        self,
        label: Optional[str] = "workflow",
        dirname: Optional[str] = "workflow",
        input_files: Optional[list] = None,
        rename_input_files: Optional[list] = None,
        workflow: Optional[Workflow] = None,
    ):
        if input_files is None:
            input_files = []
        if rename_input_files is None:
            rename_input_files = []
        if workflow is None:
            workflow = Workflow()

        # directory and dependency setting
        self.label = label
        self.dirname = dirname
        self.input_files = input_files
        self.rename_input_files = rename_input_files
        self.output_files = []
        self.output_values = {}
        self.status = "init"  # 'init', 'success', 'running', 'failure'
        self.workflow = workflow
        self.run_file = f"running_{label}"
        self.done_file = f"done_{label}"

        # project directory
        self.root_dir = os.getcwd()
        self.project_dir = os.path.join(os.getcwd(), self.dirname)

    def __preparation(self):
        logger.info(f"project dir. = {self.project_dir}")
        if os.path.isdir(self.project_dir):
            logger.info(f"eWorkflow={self.label} has been launched.")
            logger.info("Project dir. has been generated.")
            logger.info("Skip copying input files.")
            logger.info(
                "To start the workflow from scratch, plz. delete the project dir."
            )
        else:
            logger.info(f"eWorkflow={self.label} has not been launched.")
            logger.info("Creating project dir. and copying input files.")
            os.makedirs(self.project_dir, exist_ok=False)
            # copy input files
            logger.info(f"input files = {self.input_files}")
            if len(self.rename_input_files) != 0:
                assert len(self.input_files) == len(self.rename_input_files)
                rename_flag = True
            else:
                rename_flag = False
            logger.info(f"rename_flag = {rename_flag}")
            logger.info(f"cd {os.getcwd()}")
            for i, file in enumerate(self.input_files):
                if os.path.isfile(file):  # file
                    if rename_flag:
                        refile = self.rename_input_files[i]
                        shutil.copy(
                            os.path.join(file),
                            os.path.join(
                                self.project_dir, os.path.basename(refile)
                            ),
                        )
                    else:
                        shutil.copy(
                            os.path.join(file),
                            os.path.join(
                                self.project_dir, os.path.basename(file)
                            ),
                        )
                else:  # directories
                    if rename_flag:
                        refile = self.rename_input_files[i]
                        if os.path.isdir(
                            os.path.join(
                                self.project_dir, os.path.basename(refile)
                            )
                        ):
                            shutil.rmtree(
                                os.path.join(
                                    self.project_dir, os.path.basename(refile)
                                )
                            )
                        shutil.copytree(
                            os.path.join(file),
                            os.path.join(
                                self.project_dir, os.path.basename(refile)
                            ),
                        )
                    else:
                        if os.path.isdir(
                            os.path.join(
                                self.project_dir, os.path.basename(file)
                            )
                        ):
                            shutil.rmtree(
                                os.path.join(
                                    self.project_dir, os.path.basename(file)
                                )
                            )
                        shutil.copytree(
                            os.path.join(file),
                            os.path.join(
                                self.project_dir, os.path.basename(file)
                            ),
                        )

    async def async_launch(self):
        os.chdir(self.root_dir)
        self.__preparation()
        """avoid complication. This is obsolete.
        if not os.path.isfile(self.run_file) and not os.path.isfile(
            self.done_file
        ):
            logger.info(f"eWorkflow={self.label} has not been launched.")
            logger.info(f"Copying input files.")
            self.__preparation()
        else:
            logger.info(f"eWorkflow={self.label} has been launched.")
            logger.info(f"Skip copying input files.")
        if os.path.isfile(self.done_file):
            os.remove(self.done_file)
        with open(self.run_file, "w") as f:
            f.write("")
        """
        os.chdir(self.project_dir)
        (
            self.status,
            self.output_files,
            self.output_values,
        ) = await self.workflow.async_launch()
        os.chdir(self.root_dir)
        # if os.path.isfile(self.run_file): os.remove(self.run_file)
        # with open(self.done_file, "w") as f: f.write("")
        return self.status, self.output_files, self.output_values

    def launch(self):
        return asyncio.run(self.async_launch())


if __name__ == "__main__":
    from logging import getLogger

    log_level = "DEBUG"
    logger = getLogger("turboworkflow")
    logger.setLevel(log_level)
    stream_handler = StreamHandler()
    stream_handler.setLevel(log_level)
    handler_format = Formatter(
        "Module-%(name)s, LogLevel-%(levelname)s, Line-%(lineno)d %(message)s"
    )
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    logger = getLogger("Turbo-Workflows")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter(
        "%(name)s - %(levelname)s - %(lineno)d - %(message)s"
    )
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    from workflow_lrdmc import LRDMC_workflow

    # from workflow_lrdmc_ext import LRDMC_ext_workflow
    from utils_turboworkflows.turboworkflows_env import turbo_workflows_root

    os.chdir(os.path.join(turbo_workflows_root, "tests"))

    """
    clrdmcext_workflow = cWorkflow(
        label='clrdmcext-workflow',
        dirname='clrdmcext-workflow',
        input_files=["fort.10", "pseudo.dat"],
        workflow=LRDMC_ext_workflow(
                ## job
                server_machine_name="kagayaki",
                cores=64,
                openmp=1,
                queue="DEFAULT",
                version="stable",
                sleep_time=30, # sec.
                jobpkl_name="job_manager",
                ## lrdmc
                lrdmc_max_continuation=3,
                lrdmc_pkl_name="lrdmc_genius",
                lrdmc_target_error_bar=1.0e-3, # Ha
                lrdmc_trial_steps= 150,
                lrdmc_bin_block = 10,
                lrdmc_warmupblocks = 5,
                lrdmc_correcting_factor=10,
                lrdmc_trial_etry=-17.208,
                lrdmc_alat_list=[-0.40, -0.30],
                lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
                lrdmc_num_walkers = -1, # default -1 -> num of MPI process.
                lrdmc_twist_average=False,
                lrdmc_kpoints=[],
                lrdmc_force_calc_flag=False,
                lrdmc_maxtime=172000,
            )
    )
    clrdmcext_workflow.launch()
    """

    clrdmc_workflow = eWorkflow(
        label="clrdmc-workflow",
        dirname="clrdmc-workflow",
        input_files=["fort.10", "pseudo.dat"],
        workflow=LRDMC_workflow(
            # job
            server_machine_name="kagayaki",
            cores=64,
            openmp=1,
            queue="DEFAULT",
            version="stable",
            sleep_time=30,  # sec.
            jobpkl_name="job_manager",
            # lrdmc
            lrdmc_max_continuation=3,
            lrdmc_pkl_name="lrdmc_genius",
            lrdmc_target_error_bar=1.0e-3,  # Ha
            lrdmc_trial_steps=150,
            lrdmc_bin_block=10,
            lrdmc_warmupblocks=5,
            lrdmc_correcting_factor=10,
            lrdmc_trial_etry=-17.208,
            lrdmc_alat=-0.30,
            lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
            lrdmc_num_walkers=-1,  # default -1 -> num of MPI process.
            lrdmc_twist_average=False,
            lrdmc_kpoints=[],
            lrdmc_force_calc_flag=False,
            lrdmc_maxtime=172000,
        ),
    )
    clrdmc_workflow.launch()

    # moved to examples
