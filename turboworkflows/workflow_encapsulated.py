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
            logger.info(f"Encapsulated Workflow={self.label} has been launched.")
            logger.info("Project directory has been generated.")
            logger.info("Skip copying input files.")
            logger.info(
                "To start the workflow from scratch, plz. delete the project dir."
            )
        else:
            logger.info(f"Encapsulated Workflow={self.label} has not been launched.")
            logger.info("Creating project directory and copying input files.")
            os.makedirs(self.project_dir, exist_ok=False)
            # copy input files
            logger.info(f"input files = {self.input_files}")
            if len(self.rename_input_files) != 0:
                assert len(self.input_files) == len(self.rename_input_files)
                rename_flag = True
            else:
                rename_flag = False
            logger.debug(f"rename_flag = {rename_flag}")
            logger.debug(f"cwd = {os.getcwd()}")
            for i, file in enumerate(self.input_files):
                if os.path.isfile(file):  # file
                    if rename_flag:
                        refile = self.rename_input_files[i]
                        shutil.copy(
                            os.path.join(file),
                            os.path.join(self.project_dir, os.path.basename(refile)),
                        )
                    else:
                        shutil.copy(
                            os.path.join(file),
                            os.path.join(self.project_dir, os.path.basename(file)),
                        )
                else:  # directories
                    if rename_flag:
                        refile = self.rename_input_files[i]
                        if os.path.isdir(
                            os.path.join(self.project_dir, os.path.basename(refile))
                        ):
                            shutil.rmtree(
                                os.path.join(self.project_dir, os.path.basename(refile))
                            )
                        shutil.copytree(
                            os.path.join(file),
                            os.path.join(self.project_dir, os.path.basename(refile)),
                        )
                    else:
                        if os.path.isdir(
                            os.path.join(self.project_dir, os.path.basename(file))
                        ):
                            shutil.rmtree(
                                os.path.join(self.project_dir, os.path.basename(file))
                            )
                        shutil.copytree(
                            os.path.join(file),
                            os.path.join(self.project_dir, os.path.basename(file)),
                        )

    async def async_launch(self):
        os.chdir(self.root_dir)
        self.__preparation()
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
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    # moved to examples
