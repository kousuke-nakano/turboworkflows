# -*- coding: utf-8 -*-

# import python modules
import os
import time

import random
import re
import stat
import paramiko
import yaml
import shutil
import pathlib
import subprocess
from subprocess import PIPE

# define logger
from logging import getLogger, StreamHandler, Formatter

# import file-manager modules
from .file_manager_env import (
    file_manager_config_template_dir,
    file_manager_config_dir,
)

logger = getLogger("Turbo-Workflows").getChild(__name__)


class Machine:

    ssh_retry_time = 120
    ssh_retry_max_num = 10

    def __init__(self, machine):
        self.machine_info_yaml = os.path.join(
            file_manager_config_dir, "machine_data.yaml"
        )

        # check config dir exists.
        if not os.path.isdir(file_manager_config_dir):
            logger.info(
                f"{file_manager_config_dir} is not found. Probably, this is the first run."
            )
            os.makedirs(file_manager_config_dir, exist_ok=True)
            shutil.copytree(file_manager_config_template_dir, file_manager_config_dir)
            logger.info(f"{file_manager_config_dir} has been generated.")
            logger.info(f"Please edit {self.machine_info_yaml}")
            raise FileNotFoundError

        # open data files
        try:
            with open(self.machine_info_yaml, "r") as yf:
                self.data = yaml.safe_load(yf)[machine]
                # logger.debug(self.data)
        except FileNotFoundError:
            logger.error(f"The yaml file={self.machine_info_yaml} is not found!!")
            raise FileNotFoundError
        except KeyError:
            logger.error(f"machine={machine} is not defined in the database!!")
            logger.error("Plz. edit the following file according to the template.")
            logger.error(self.machine_info_yaml)
            raise KeyError

        self.__name = machine
        logger.debug(self.machine_type)

        self.ssh_status = False

    def ssh_open(self):
        if self.machine_type == "remote":
            if not self.ssh_status:
                rw = random.randint(1, 4)
                logger.info(
                    f"wait {rw} secs before opening a ssh connection to the remote machine."
                )
                time.sleep(rw)
                logger.info("A ssh connection is open via paramiko module.")
                ssh_config = paramiko.SSHConfig()
                try:
                    config_file = os.path.join(os.getenv("HOME"), ".ssh/config")
                    ssh_config.parse(open(config_file, "r"))
                except FileNotFoundError:
                    logger.error(
                        f"TurboWorkflows needs the ssh config file ({config_file})"
                    )
                lkup = ssh_config.lookup(self.__name)

                hostname = lkup["hostname"]
                username = lkup["user"]
                key_filename = lkup["identityfile"]

                logger.debug(f"paramiko ssh hostname = {hostname}")
                logger.debug(f"paramiko ssh username = {username}")
                logger.debug(f"paramiko ssh key_filename = {key_filename}")

                try:
                    proxy_command = lkup["proxycommand"]
                    logger.debug(f"paramiko ssh proxy-command = {proxy_command}")
                    proxy_flag = True
                except KeyError:
                    proxy_flag = False

                self.username = username
                self.ssh = paramiko.SSHClient()
                self.ssh.load_system_host_keys()
                for tt in range(self.ssh_retry_max_num):
                    try:
                        if proxy_flag:
                            self.ssh.connect(
                                hostname=hostname,
                                username=username,
                                key_filename=key_filename,
                                sock=paramiko.ProxyCommand(proxy_command),
                            )
                        else:
                            self.ssh.connect(
                                hostname=hostname,
                                username=username,
                                key_filename=key_filename,
                            )
                        logger.info(f"ssh using paramiko is successful.")
                        break
                    except paramiko.ssh_exception.SSHException:
                        logger.warning(
                            f"ssh using paramiko failed. Wait {self.ssh_retry_time} sec before the next trial."
                        )
                        time.sleep(self.ssh_retry_time)

                    if tt == self.ssh_retry_max_num - 1:
                        logger.error(
                            f"ssh using paramiko failed in all {self.ssh_retry_max_num}-times ssh trials"
                        )
                        raise paramiko.SSHException

                self.sftp = self.ssh.open_sftp()
                self.ssh_status = True

            else:
                logger.info(
                    "The ssh connection is already established using paramiko module."
                )
                logger.debug(f"self.ssh_status = {self.ssh_status}")

    def ssh_close(self):
        if self.machine_type == "remote":
            logger.debug(f"self.ssh_status = {self.ssh_status}")
            if self.ssh_status:
                logger.info("The ssh connection is close via paramiko module.")
                self.ssh.close()
                self.sftp.close()
                del self.ssh
                del self.sftp
                self.ssh_status = False

    def __str__(self):

        output = [f"Machine obj. {self.name}"]
        return "\n".join(output)

    def get_value(self, key):
        try:
            return self.data[key]
        except KeyError:
            logger.error(f"{key} key is not defined in the database!!")
            logger.error("Plz. edit the following file according to the template.")
            logger.error(self.machine_info_yaml)
            raise KeyError

    @property
    def name(self):
        return self.__name

    @property
    def machine_type(self):
        key = "machine_type"
        value = self.get_value(key=key)
        if value not in {"local", "remote"}:
            logger.error(f"value = {value}")
            logger.error("value should be local or remote")
            raise ValueError
        return value

    @property
    def ip(self):
        key = "ip"
        return self.get_value(key=key)

    @property
    def file_manager_root(self):
        key = "file_manager_root"
        return self.get_value(key=key)

    @property
    def queuing(self):
        key = "queuing"
        return self.get_value(key=key)

    @property
    def computation(self):
        key = "computation"
        return self.get_value(key=key)

    @property
    def jobsubmit(self):
        key = "jobsubmit"
        return self.get_value(key=key)

    @property
    def jobcheck(self):
        key = "jobcheck"
        return self.get_value(key=key)

    @property
    def jobdel(self):
        key = "jobdel"
        return self.get_value(key=key)

    @property
    def jobnum_index(self):
        key = "jobnum_index"
        return self.get_value(key=key)

    def get_job_list(self):
        command = f"{self.jobcheck}"
        stdout, stderr = self.run_command(command)
        return stdout, stderr

    def get_job_list_as_text(self):
        stdout, stderr = self.get_job_list()
        return stdout.split("\n")

    def delete_job(self, jobid):
        command = f"{self.jobdel} {jobid}"
        stdout, stderr = self.run_command(command)
        return stdout.split("\n")

    def run_command(self, command, execute_dir=None):
        trial_num = 10
        jjj = 0
        while True:
            if execute_dir is None:
                command_r = f"{command}"
            else:
                if self.machine_type == "remote":
                    self.ssh_open()
                    fileattr = self.sftp.lstat(execute_dir)
                    if not stat.S_ISDIR(fileattr.st_mode):
                        logger.error(
                            f"{execute_dir} is not found on the remote machine."
                        )
                        raise FileNotFoundError
                else:
                    if not os.path.isdir(execute_dir):
                        logger.error(f"{execute_dir} is not found.")
                        raise FileNotFoundError

                command_r = f"cd {execute_dir}; {command}"

            # command_r=timeout_command(command=command_r)
            logger.debug(f"command = {command_r} in run_command")

            if self.machine_type == "local":
                for ii in range(3):
                    try:
                        proc = subprocess.run(
                            command_r,
                            shell=True,
                            stdout=PIPE,
                            stderr=PIPE,
                            text=True,
                            timeout=1200,
                        )
                        logger.warning(
                            f"subprocess is successful (ii={ii}). break the loop"
                        )
                        exit_status, stdout, stderr = (
                            proc.returncode,
                            proc.stdout,
                            proc.stderr,
                        )
                        logger.debug(f"exit_status={exit_status}")
                        break
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"subprocess is timeout (ii={ii}). iterate the loop."
                        )
                        exit_status = 99
                    logger.warning("wait 60 secs.")
                    time.sleep(60)

                if exit_status == 0:
                    # success run_command
                    logger.debug(f"stdout = {stdout}")
                    logger.debug(f"stderr = {stderr}")
                    logger.debug(f"exit_status = {exit_status}")
                    break
                else:
                    # failure run_command
                    logger.warning(f"stdout = {stdout}")
                    logger.warning(f"stderr = {stderr}")
                    logger.warning(f"exit_status = {exit_status}")
                    logger.warning(f"command={command_r} did not work.")
                    logger.warning(
                        f"The command will be retried after {self.ssh_retry_time}s sleep."
                    )
                    time.sleep(self.ssh_retry_time)
                if jjj > trial_num:
                    break
                jjj += 1

                if jjj > trial_num:
                    logger.error("Something wrong in run_command!!")
                    raise ValueError

            else:  # remote
                self.ssh_open()
                logger.debug(f"command_r={command_r}")
                _, pstdout, pstderr = self.ssh.exec_command(command=command_r)
                exit_status = pstdout.channel.recv_exit_status()
                logger.debug(f"exit_status = {exit_status}")
                stdout, stderr = str(pstdout.read().decode("utf-8").strip()), str(
                    pstderr.read().decode("utf-8").strip()
                )

                if exit_status == 0:
                    logger.debug(f"command_r={command_r} was successful.")
                    logger.debug(f"stdout={stdout}")
                    logger.debug(f"stderr={stderr}")
                    break
                else:
                    logger.error(f"command_r={command_r} failed.")
                    logger.error(f"stdout={stdout}")
                    logger.error(f"stderr={stderr}")
                    raise ValueError

        return stdout, stderr

    def is_file(self, file_name):
        if not pathlib.Path(file_name).is_absolute():
            logger.error(f"file_name={file_name} is not an absolute path.")

        if self.machine_type == "local":
            if os.path.isfile(file_name):
                return True
            else:
                return False
        else:
            self.ssh_open()
            try:
                fileattr = self.sftp.lstat(file_name)
                if stat.S_ISREG(fileattr.st_mode):
                    return True
                else:
                    return False
            except IOError:
                return False

    def is_dir(self, dir_name):
        logger.debug(f"check if dir={dir_name} exists.")
        if not pathlib.Path(dir_name).is_absolute():
            logger.error(f"dir_name={dir_name} is not an absolute path.")

        if self.machine_type == "local":
            if os.path.isdir(dir_name):
                return True
            else:
                return False
        else:
            self.ssh_open()
            fileattr = self.sftp.lstat(dir_name)
            if stat.S_ISDIR(fileattr.st_mode):
                return True
            else:
                return False

    def exist(self, object_name):
        logger.debug(f"check if file or dir={object_name} exists on {self.name}.")
        if not pathlib.Path(object_name).is_absolute():
            logger.error(f"dir_name={object_name} is not an absolute path.")

        if self.machine_type == "local":
            if os.path.exists(object_name):
                return True
            else:
                return False
        else:
            self.ssh_open()
            try:
                fileattr = self.sftp.lstat(object_name)
                if stat.S_ISDIR(fileattr.st_mode):
                    return True
                elif stat.S_ISREG(fileattr.st_mode):
                    return True
                else:
                    return False
            except FileNotFoundError:
                return False

    def is_alive(self):
        # logger.info("wait 1 sec.")
        # time.sleep(1)
        return True


class Machines_handler:

    def __init__(self, client_machine_name, server_machine_name):

        self.client_machine = Machine(client_machine_name)
        self.server_machine = Machine(server_machine_name)

    def ssh_close(self):
        self.client_machine.ssh_close()
        self.server_machine.ssh_close()

    # data transfer class
    def put(self, from_file, to_file, exclude_patterns=[]):
        self.object_transfer(
            from_machine=self.client_machine,
            from_object=from_file,
            to_machine=self.server_machine,
            to_object=to_file,
            exclude_patterns=exclude_patterns,
            dir_transfer=False,
        )

    def put_dir(self, from_dir, to_dir, exclude_patterns=[]):
        self.object_transfer(
            from_machine=self.client_machine,
            from_object=from_dir,
            to_machine=self.server_machine,
            to_object=to_dir,
            exclude_patterns=exclude_patterns,
            dir_transfer=True,
        )

    def get(self, from_file, to_file, exclude_patterns=[]):
        self.object_transfer(
            from_machine=self.server_machine,
            from_object=from_file,
            to_machine=self.client_machine,
            to_object=to_file,
            exclude_patterns=exclude_patterns,
            dir_transfer=False,
        )

    def get_dir(self, from_dir, to_dir, exclude_patterns=[]):
        self.object_transfer(
            from_machine=self.server_machine,
            from_object=from_dir,
            to_machine=self.client_machine,
            to_object=to_dir,
            exclude_patterns=exclude_patterns,
            dir_transfer=True,
        )

    def get_sftp_file(self, source, target, exclude_patterns=[]):
        """Download the contents of the source file to the target path."""
        if not any([re.match(p, os.path.basename(source)) for p in exclude_patterns]):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            self.server_machine.ssh_open()
            sftp = self.server_machine.sftp
            sftp.get(source, target)

    def put_sftp_file(self, source, target, exclude_patterns=[]):
        """Uploads the contents of the source file to the target path."""
        if not any([re.match(p, os.path.basename(source)) for p in exclude_patterns]):
            # self.server_machine.ssh.exec_command(f"mkdir -p {os.path.dirname(target)}")
            self.server_machine.run_command(f"mkdir -p {os.path.dirname(target)}")
            self.server_machine.ssh_open()
            sftp = self.server_machine.sftp
            sftp.put(source, target)

    def get_sftp_dir(self, source, target, exclude_patterns=[]):
        """Download the contents of the source directory to the target path. The
        target directory needs to exists. All subdirectories in source are
        created under target.
        """
        os.makedirs(target, exist_ok=True)
        self.server_machine.ssh_open()
        sftp = self.server_machine.sftp

        for item in sftp.listdir_attr(source):
            if any([re.match(p, os.path.basename(item)) for p in exclude_patterns]):
                continue
            logger.debug(f" transfered file or dir = {item}")
            fileattr = sftp.lstat(os.path.join(source, item))
            if stat.S_IFREG(fileattr.st_mode):
                sftp.get(os.path.join(source, item), os.path.join(target, item))
            else:
                os.makedirs("%s/%s" % (target, item), exists_ok=True)
                self.get_sftp_dir(
                    os.path.join(source, item), os.path.join(target, item)
                )

    def put_sftp_dir(self, source, target, exclude_patterns=[]):
        """Uploads the contents of the source directory to the target path. The
        target directory needs to exists. All subdirectories in source are
        created under target.
        """
        self.server_machine.run_command(f"mkdir -p {target}")
        self.server_machine.ssh_open()
        sftp = self.server_machine.sftp
        for item in os.listdir(source):
            if any([re.match(p, os.path.basename(item)) for p in exclude_patterns]):
                continue
            logger.debug(f" transfered file or dir = {item}")
            if os.path.isfile(os.path.join(source, item)):
                sftp.put(os.path.join(source, item), os.path.join(target, item))
            else:
                # self.sftp_mkdir("%s/%s" % (target, item), ignore_existing=True)
                # self.server_machine.ssh.exec_command(
                #    f"mkdir -p {os.path.join(target, item)}"
                # )
                self.put_sftp_dir(
                    os.path.join(source, item),
                    os.path.join(target, item),
                    exclude_patterns=exclude_patterns,
                )

    '''
    def sftp_mkdir(self, path, mode=511, ignore_existing=False):
        self.server_machine.ssh_open()
        sftp = self.server_machine.sftp
        """Augments mkdir by adding an option to not fail if the folder exists"""
        try:
            sftp.mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise
    '''

    # core object transfer method
    def object_transfer(
        self,
        from_machine,
        from_object,
        to_machine,
        to_object,
        exclude_patterns=[],
        dir_transfer=False,
    ):
        # time.sleep(3)
        # check
        if not pathlib.Path(from_object).is_absolute():
            logger.error(f"from_object = {from_object} is not an absolute path")
            raise ValueError
        if not pathlib.Path(to_object).is_absolute():
            logger.error(f"to_object = {to_object} is not an absolute path")
            raise ValueError

        logger.debug(f"makedir {os.path.dirname(to_object)} on {to_machine.name}")
        to_dir = os.path.dirname(to_object)
        command = f"mkdir -p {to_dir}"
        to_machine.run_command(command)

        if not to_machine.is_dir(dir_name=to_dir):
            logger.error(f"{to_dir} is not created.")
            raise FileNotFoundError

        if from_machine.machine_type == "local" and to_machine.machine_type == "local":
            logger.debug("No data transfer is needed.")
        elif (
            from_machine.machine_type == "local" and to_machine.machine_type == "remote"
        ) or (
            from_machine.machine_type == "remote" and to_machine.machine_type == "local"
        ):
            logger.info(f"From:: {from_object}")
            logger.info(f"To:: {to_object}")

            # file transfer
            if (
                from_machine.machine_type == "local"
                and to_machine.machine_type == "remote"
            ):  # local -> remote
                logger.info(
                    f"Transfer data from local machine ({from_machine.name}) to remote machine ({to_machine.name}) using paramiko."
                )
                if dir_transfer:  # dir
                    self.put_sftp_dir(
                        from_object, to_object, exclude_patterns=exclude_patterns
                    )
                else:  # file
                    self.put_sftp_file(
                        from_object, to_object, exclude_patterns=exclude_patterns
                    )

            else:  # remote -> local
                logger.info(
                    f"Transfer data from remote machine ({from_machine.name}) to local machine ({to_machine.name}) using paramiko."
                )
                if dir_transfer:  # dir
                    self.get_sftp_dir(
                        from_object, to_object, exclude_patterns=exclude_patterns
                    )
                else:  # file
                    self.get_sftp_file(
                        from_object, to_object, exclude_patterns=exclude_patterns
                    )

        else:
            raise NotImplementedError


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

    machine_handler_test_dir = os.path.join(file_manager_test_dir, "machines_handler")
    os.chdir(machine_handler_test_dir)

    remote_user = "maezono"
    remote_ip = "amd"  # IP or alias in .ssh/config
    remote_port = 22
    ssh_key = os.path.expanduser("~/.ssh/id_rsa")

    m_handler = Machines_handler(
        client_machine_name="localhost", server_machine_name="amd"
    )
    l_handler = Machines_handler(
        client_machine_name="kagayaki", server_machine_name="amd"
    )

    local_test_dir = machine_handler_test_dir
    amd_test_dir = "/work/nkousuke/filemanager_test"
    kagayaki_test_dir = "/home/nkousuke/lustre/toBeSync/turboRVB/03turbo-genius-development/file-manager/tests/job_manager"
    """
    assert m_handler.server_is_file(file_name="/work/nkousuke/filemanager_test/README_remote.md")
    assert not m_handler.server_is_file(file_name="/work/nkousuke/filemanager_test/README_remote.m")
    assert m_handler.server_is_dir(dir_name="/work/nkousuke/filemanager_test/readme_remote_dir")
    assert not m_handler.server_is_dir(dir_name="/work/nkousuke/filemanager_test/readme_remote_di")
    assert m_handler.server_exist(object_name="/work/nkousuke/filemanager_test/readme_remote_dir")
    assert not m_handler.server_exist(object_name="/work/nkousuke/filemanager_test/readme_remote_di")

    assert m_handler.client_is_file(file_name=os.path.join(machine_handler_test_dir,"README_remote.md"))
    assert not m_handler.client_is_file(file_name=os.path.join(machine_handler_test_dir,"README_remote.m"))
    assert m_handler.client_is_dir(dir_name=os.path.join(machine_handler_test_dir,"readme_remote_dir"))
    assert not m_handler.client_is_dir(dir_name=os.path.join(machine_handler_test_dir,"readme_remote_di"))
    assert m_handler.client_exist(object_name=os.path.join(machine_handler_test_dir,"readme_remote_dir"))
    assert not m_handler.client_exist(object_name=os.path.join(machine_handler_test_dir,"readme_remote_di"))
    """

    """
    m_handler.put(
        from_file=os.path.join(local_test_dir, "README.md"),
        to_file=os.path.join(amd_test_dir, "README.md")
    )

    l_handler.put(
        from_file=os.path.join(kagayaki_test_dir, "file-manager.o3973276"),
        to_file=os.path.join(amd_test_dir, "file-manager.o3973276")
    )
    """

    m_handler.put_dir(
        from_dir=os.path.join(local_test_dir, "readme_dir"),
        to_dir=os.path.join(amd_test_dir, "readme_dir"),
    )
    l_handler.put_dir(
        from_dir=os.path.join(kagayaki_test_dir, "text_on_kagayaki"),
        to_dir=os.path.join(amd_test_dir, "text_on_kagayaki"),
    )

    """

    if os.path.isfile("README_remote.md"): os.remove("README_remote.md")
    sftp.get(
        from_file="/work/nkousuke/filemanager_test/README_remote.md",
        to_file=os.path.join(sftp_handler_test_dir, "README_remote.md")
    )

    if os.path.isdir("readme_remote_dir"): shutil.rmtree("readme_remote_dir")
    sftp.get_dir(
        from_dir="/work/nkousuke/filemanager_test/readme_remote_dir",
        to_dir=os.path.join(sftp_handler_test_dir, "readme_remote_dir")
    )

    """
