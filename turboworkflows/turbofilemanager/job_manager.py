# -*- coding: utf-8 -*-
import os

import pickle
import shutil
import yaml
import toml
import re
import time
from datetime import datetime
from collections import OrderedDict
from logging import getLogger, StreamHandler, Formatter

# file-manager related path lists
from .file_manager_env import file_manager_config_dir, file_manager_config_template_dir
from .data_transfer_manager import Machine, Data_transfer

yaml.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: OrderedDict(loader.construct_pairs(node)),
)

logger = getLogger("Turbo-Workflows").getChild(__name__)


class Job_submission:
    stat_time_sleep = 60  # sec.

    def __init__(
        self,
        client_machine_name,
        server_machine_name,
        # package related
        package,
        binary=None,
        version=None,
        mpi=False,
        input_file=None,
        output_file="out.o",
        preoption=None,
        postoption=None,
        input_redirect=True,
        # job resources related
        queue_label="default",
        # job information related
        jobname="file-manager",
        pkl_name="job_manager.pkl",
        safe_mode=False,
    ):

        self.client_machine = Machine(client_machine_name)
        self.server_machine = Machine(server_machine_name)

        self.data_transfer = Data_transfer(
            client_machine_name=client_machine_name,
            server_machine_name=server_machine_name,
        )

        self.queue_label = queue_label
        self.mpi = mpi

        if not self.server_machine.computation:
            logger.error("The server machine is not for computations!!!")
            raise ValueError

        # package and cores
        self.package = package
        logger.info(f"package={self.package}")

        # check config dir exists.
        if not os.path.isdir(file_manager_config_dir):
            logger.info(f"{file_manager_config_dir} is not found.")
            #os.makedirs(file_manager_config_dir, exist_ok=True)
            shutil.copytree(file_manager_config_template_dir, file_manager_config_dir)
            logger.info(f"{file_manager_config_dir} has been generated.")
            logger.info(
                f"Please edit directories and files in {file_manager_config_dir}"
            )
            raise ValueError

        # open data files
        try:
            with open(
                os.path.join(
                    file_manager_config_dir,
                    self.server_machine.name,
                    "package.yaml",
                ),
                "r",
            ) as yf:
                data = yaml.safe_load(yf)
                self.package_data = data[package]
        except FileNotFoundError:
            logger.error(
                f"{os.path.join(file_manager_config_dir, self.server_machine.name, 'package.yaml')} is not found!!"
            )
            raise FileNotFoundError

        # version
        if version is None:
            logger.warning("version is not specified.")
            self.version = list(self.package_data["binary_path"].keys())[0]
            self.binary_path = list(self.package_data["binary_path"].values())[0]
            logger.warning(f"default version is {self.version}")
            logger.warning(f"default binary_path is {self.binary_path}")

        else:
            if version not in self.package_data["binary_path"].keys():
                logger.error(
                    f"version={version} does not exist in binary_path. Plz. check package.yaml"
                )
                raise KeyError
            self.version = version
            self.binary_path = self.package_data["binary_path"][version]

        # binary
        if binary is None:
            logger.warning("binary is not specified.")
            self.binary = self.package_data["binary_list"][0]
            logger.warning(f"default binary is {self.binary}")
        else:
            if binary not in self.package_data["binary_list"]:
                logger.error(f"binary={binary}")
                logger.error(f"binary_list={self.package_data['binary_list']}")
                raise KeyError
            self.binary = binary

        # queue data
        self.queue_data_toml = os.path.join(
            file_manager_config_dir,
            self.server_machine.name,
            "queue_data.toml",
        )
        try:
            dict_toml = toml.load(open(self.queue_data_toml))
        except FileNotFoundError:
            logger.error(f"{self.queue_data_toml} is not found!!")
            raise FileNotFoundError

        try:
            self.queue_data = dict_toml[queue_label]
        except KeyError:
            logger.error(
                f"queue_label = {queue_label} is not found in {self.queue_data_toml}."
            )

        # check job template
        if mpi:
            try:
                self.job_submission_template = self.package_data["job_template"]["mpi"]
            except KeyError:
                logger.error("mpi is not defined in job_template.")
                logger.error(f"Please check {self.queue_data_toml}")
                raise KeyError
        else:
            try:
                self.job_submission_template = self.package_data["job_template"][
                    "nompi"
                ]
            except KeyError:
                logger.error("mpi is not defined in job_template.")
                logger.error(f"Please check {self.queue_data_toml}")
                raise KeyError

        # other input information
        self.jobname = jobname
        self.preoption = preoption
        self.postoption = postoption
        self.input_file = input_file
        self.output_file = output_file
        self.pkl_name = pkl_name
        self.safe_mode = safe_mode
        self.input_redirect = input_redirect

        # job information!!
        try:
            self.max_job_submit = self.queue_data["max_job_submit"]
        except KeyError:
            logger.warning(
                "max_job_submit is not defined in queue_data.toml. set 1000."
            )
            self.max_job_submit = 1000
        self.job_number = None  # job ID.
        self.job_running = False  # 0: end, 1 running.
        self.job_dir = None
        self.job_submit_date = None
        self.job_check_last_time = None
        self.job_fetch_date = None

    def generate_script(self, submission_script="submit.sh"):

        def replaced_lines(lines, keyword, value):
            buffer = [line for line in lines if re.match(f".*{keyword}.*", line)]
            if len(buffer) == 0:
                return lines
            else:
                # assert len(buffer) == 1 # to be refactored.
                # buffer = buffer[0]
                for buf in buffer:
                    buffer_index = lines.index(buf)
                    lines[buffer_index] = lines[buffer_index].replace(
                        keyword.replace("\\", ""), str(value)
                    )
                return lines

        # read the template
        with open(
            os.path.join(
                file_manager_config_dir,
                self.server_machine.name,
                self.job_submission_template,
            ),
            "r",
        ) as f:
            lines = f.readlines()

        # Replacing keywords in submit_script
        # [input and output]
        if self.input_file is None:
            lines = replaced_lines(lines, " < \$INPUT", "")
        else:
            lines = replaced_lines(lines, "_INPUT_", self.input_file)
            if not self.input_redirect:
                lines = replaced_lines(lines, " < \$INPUT", " $INPUT")
        lines = replaced_lines(lines, "_OUTPUT_", self.output_file)
        # [preoption]
        if self.preoption is None:
            lines = replaced_lines(lines, "\$PREOPTION", "")
        else:
            lines = replaced_lines(lines, "_PREOPTION_", '"' + self.preoption + '"')
        # [postoption]
        if self.postoption is None:
            lines = replaced_lines(lines, "\$POSTOPTION", "")
        else:
            lines = replaced_lines(lines, "_POSTOPTION_", '"' + self.postoption + '"')
        # [BINARY_ROOT] and [BINARY]
        if self.binary_path is None:
            lines = replaced_lines(lines, "_BINARY_ROOT_/", "")
        else:
            lines = replaced_lines(lines, "_BINARY_ROOT_", self.binary_path)
        lines = replaced_lines(lines, "_BINARY_", self.binary)
        # [JOB_NAME]
        lines = replaced_lines(lines, "_JOBNAME_", self.jobname)

        # other values defined in queue_data.toml
        logger.info(
            f"Variables defined in {self.queue_data_toml} are {['_'+key.upper()+'_' for key in self.queue_data.keys()]} with label={self.queue_label}."
        )

        for key, value in self.queue_data.items():
            if key != "mpi":
                lines = replaced_lines(lines, str("_" + key.upper() + "_"), value)

        # write the replaced job submission script.
        with open(submission_script, "w") as f:
            f.writelines(lines)

        with open(self.pkl_name, "wb") as f:
            pickle.dump(self, f)

    def job_submit(self, submission_script="submit.sh", from_objects=[]):
        if not self.jobnum_check():
            logger.info("The current num. job exceeds max")
            self.job_submit_date = None
            self.job_number = None
            self.job_running = False
            return False, self.job_number
        else:
            try:
                logger.debug("The computational node is available")
                command = f"{self.server_machine.jobsubmit} {submission_script}"

                client_home = self.client_machine.file_manager_root
                server_home = self.server_machine.file_manager_root
                if self.safe_mode:
                    if not self.client_machine.is_dir(client_home):
                        logger.error(f"{client_home} is not found.")
                        raise FileNotFoundError
                    if not self.server_machine.is_dir(server_home):
                        logger.error(f"{server_home} is not found.")
                        raise FileNotFoundError
                local_current_dir = os.path.abspath(os.getcwd())

                if (
                    self.client_machine.machine_type == "local"
                    and self.server_machine.machine_type == "local"
                ):
                    server_dir = local_current_dir
                else:
                    if client_home not in local_current_dir:
                        logger.error(
                            "server-client_manager.py works only in the local_home dir."
                        )
                        raise ValueError
                    else:
                        # client_dir = local_current_dir.replace(client_home, client_home)
                        server_dir = local_current_dir.replace(client_home, server_home)

                        # data transfer
                        self.data_transfer.put_objects(from_objects=from_objects)

                if self.server_machine.queuing:
                    logger.debug("queueing system")
                    (
                        stdout,
                        stderr,
                    ) = self.server_machine.run_command(
                        command=command, execute_dir=server_dir
                    )
                    logger.debug(f"stderr={stderr.split()}")
                    if not stdout:
                        logger.error(f"Empty stdout on command!!")
                        logger.error(f'stdout = {stdout}')
                        logger.error(f'stderr = {stderr}')
                    self.job_number = stdout.strip().split()[
                        self.server_machine.jobnum_index
                    ]
                    self.job_running = True
                    self.job_dir = server_dir
                    self.job_submit_date = datetime.today()
                    logger.info(
                        f"Job submission is successful with job_number = {self.job_number}."
                    )
                else:
                    self.server_machine.run_command(
                        command=command, execute_dir=server_dir
                    )
                    self.job_number = None
                    self.job_running = False
                    self.job_dir = server_dir
                    self.job_submit_date = datetime.today()
                    logger.info("Job submission is successful.")

                self.client_machine.ssh_close()
                self.server_machine.ssh_close()
                self.data_transfer.ssh_close()
                with open(self.pkl_name, "wb") as f:
                    pickle.dump(self, f)

                return True, self.job_number

            except ValueError:
                self.job_number = None
                self.job_running = False
                logger.error("Something wrong in job_submit!!")

    def jobcheck(self):

        self.job_check_last_time = datetime.today()

        if self.server_machine.queuing:
            # if self.job_running:
            trial_num = 10
            jjj = 0
            while True:
                job_list = self.server_machine.get_job_list_as_text()
                if not job_list == "":
                    break
                if jjj > trial_num:
                    break
                logger.warning(f"{self.server_machine.jobcheck} command did not work.")
                logger.warning(
                    f"The command will be retried after {self.stat_time_sleep}s sleep."
                )
                time.sleep(self.stat_time_sleep)
                jjj += 1
            if job_list == "" and jjj > trial_num:
                logger.error("Something wrong in jobcheck!!")
                raise ValueError
            logger.info(self.job_number)
            bool_list = [
                True if re.match(f".*{self.job_number}.*", line) else False
                for line in job_list
            ]
            if any(bool_list):
                logger.info(f"job {self.job_number} is running.")
                self.job_running = True
                flag = True
            else:
                logger.info(f"job {self.job_number} is done.")
                self.job_running = False
                flag = False
            # else:
            #    flag = False
        else:
            flag = False

        self.client_machine.ssh_close()
        self.server_machine.ssh_close()
        self.data_transfer.ssh_close()
        with open(self.pkl_name, "wb") as f:
            pickle.dump(self, f)

        return flag

    def jobnum_check(self):
        if self.server_machine.queuing:
            job_list = self.server_machine.get_job_list_as_text()
            bool_list = [
                (
                    True
                    if re.match(
                        f".*\s{self.server_machine.username}\s.*\s{self.queue_data['queue']}\s.*",
                        line,
                    )  # PBS case
                    or re.match(
                        f".*\s{self.queue_data['queue']}\s.*\s{self.server_machine.username}\s.*",
                        line,
                    )  # Slurm case
                    else False
                )
                for line in job_list
            ]
            num = bool_list.count(True)
            logger.info(f"{num} jobs are running on {self.server_machine.name}")
            if num < self.max_job_submit:
                logger.info(f"{num} < max_job_submit:{self.max_job_submit}")
                flag = True
            else:
                logger.info(f"{num} >= max_job_submit:{self.max_job_submit}")
                flag = False
        else:
            flag = True

        self.client_machine.ssh_close()
        self.server_machine.ssh_close()
        self.data_transfer.ssh_close()
        with open(self.pkl_name, "wb") as f:
            pickle.dump(self, f)

        return flag

    def fetch_job(self, from_objects=[], exclude_patterns=[]):
        client_home = self.client_machine.file_manager_root
        server_home = self.server_machine.file_manager_root
        if self.safe_mode:
            if not self.client_machine.is_dir(client_home):
                logger.error(f"{client_home} is not found.")
                raise FileNotFoundError
            if not self.server_machine.is_dir(server_home):
                logger.error(f"{server_home} is not found.")
                raise FileNotFoundError
        local_current_dir = os.path.abspath(os.getcwd())

        if (
            self.client_machine.machine_type == "local"
            and self.server_machine.machine_type == "local"
        ):
            server_dir = local_current_dir

        else:

            if client_home not in local_current_dir:
                logger.error(
                    "server-client_manager.py works only in the local_home dir."
                )
                logger.error(f"client_home={client_home} is not in")
                logger.error(f"local_current_dir={local_current_dir}")
                raise ValueError

            else:
                client_dir = local_current_dir.replace(client_home, client_home)
                server_dir = local_current_dir.replace(client_home, server_home)
                logger.info(client_dir)
                logger.info(server_dir)

                # data transfer
                self.data_transfer.get_objects(
                    from_objects=from_objects, exclude_patterns=exclude_patterns
                )

        self.job_fetch_date = datetime.today()

        self.client_machine.ssh_close()
        self.server_machine.ssh_close()
        self.data_transfer.ssh_close()
        with open(self.pkl_name, "wb") as f:
            pickle.dump(self, f)

    def delete_job(self):
        # job delete
        self.server_machine.delete_job(jobid=self.job_number)
        self.job_running = False

        self.client_machine.ssh_close()
        self.server_machine.ssh_close()
        self.data_transfer.ssh_close()
        with open(self.pkl_name, "wb") as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    from logging import getLogger
    from file_manager_env import file_manager_test_dir

    log_level = "DEBUG"
    logger = getLogger("file-manager")
    logger.setLevel(log_level)
    stream_handler = StreamHandler()
    stream_handler.setLevel(log_level)
    handler_format = Formatter(
        "Module-%(name)s, LogLevel-%(levelname)s, Line-%(lineno)d %(message)s"
    )
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    job_manager_test_dir = os.path.join(file_manager_test_dir, "job_manager")
    os.chdir(job_manager_test_dir)

    job = Job_submission(
        local_machine_name="localhost",
        client_machine_name="localhost",
        server_machine_name="kagayaki",
        package="turborvb",
        cores=1536,
        openmp=1,
        queue="LARGE",
    )
    # job.generate_script(submission_script="submit.sh")
    # job.job_submit(submission_script="submit.sh")
    job.jobnum_check()
    # job.jobcheck()
