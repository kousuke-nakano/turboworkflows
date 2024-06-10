# -*- coding: utf-8 -*-

# import python modules
import os
import argparse
import yaml
import shutil
from datetime import datetime
from subprocess import call

# define logger
from logging import getLogger

# import file-manager modules
from .file_manager_env import (
    machine_handler_env_dir,
    file_manager_config_dir,
    machine_handler_env_template_dir,
)
from .data_transfer_manager import Data_transfer

try:
    from .._version import (
        version as turbofilemanager_version,
    )
except (ModuleNotFoundError, ImportError):
    turbofilemanager_version = "unknown"

logger = getLogger("Turbo-Workflows").getChild(__name__)


def file_manager_cli():
    machine_info_yaml = os.path.join(machine_handler_env_dir, "machine_data.yaml")
    try:
        with open(machine_info_yaml, "r") as yf:
            machine_list = yaml.safe_load(yf).keys()
    except FileNotFoundError:
        print(f"The yaml file={machine_info_yaml} is not found!!")
        # check config dir exists.
        if not os.path.isdir(machine_handler_env_dir):
            print(
                f"{machine_handler_env_dir} is not found. Probably, this is the first run."
            )
            os.makedirs(file_manager_config_dir, exist_ok=True)
            shutil.copytree(machine_handler_env_template_dir, machine_handler_env_dir)
            print(f"{machine_handler_env_dir} has been generated.")
            print(f"plz. edit {machine_info_yaml}")
            return
        else:
            raise FileNotFoundError

    job_list = ["put", "get", "medit"]

    # define the parser
    parser = argparse.ArgumentParser(
        epilog=f"turbo-filemanager {turbofilemanager_version}",
        usage="see [https://github.com/kousuke-nakano/turbofilemanager]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Job type:
    parser.add_argument("job", help=f"Specify job type {job_list}", choices=job_list)
    # Setting files:
    parser.add_argument(
        "-editor",
        "--editor",
        help="Choose an editor to edit the machine and queue system files",
        type=str,
        default="vim",
    )
    # Local machine
    parser.add_argument(
        "-l",
        "--local_machine",
        help=f"local machine {machine_list}",
        choices=machine_list,
        default="localhost",
    )
    # Client (typically, local machine)
    parser.add_argument(
        "-c",
        "--client_machine",
        help=f"client machine (typically local) {machine_list}",
        choices=machine_list,
        default="localhost",
    )
    # Server
    parser.add_argument(
        "-s",
        "--server_machine",
        help=f"server machine {machine_list}",
        choices=machine_list,
        default="localhost",
    )
    # Include, Exclude
    parser.add_argument(
        "-inc",
        "--include",
        help="specify an rsync include list",
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "-exc",
        "--exclude",
        help="specify an rsync exclude list",
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "-n",
        "--dryrun",
        help="dry run of rsync",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--delete",
        help="Caution!! destructive rsync operation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-safe",
        "--safe_mode",
        help="Do sanity check, e.g., the existence of target dirs.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-log", "--log_level", choices=["DEBUG", "INFO"], default="INFO"
    )

    # parse the input values
    args = parser.parse_args()

    from logging import getLogger, StreamHandler, Formatter

    logger = getLogger("file-manager")
    logger.setLevel(args.log_level)
    stream_handler = StreamHandler()
    stream_handler.setLevel(args.log_level)
    if args.log_level == "DEBUG":
        handler_format = Formatter("%(name)s l-%(lineno)d %(message)s")
    else:
        handler_format = Formatter("%(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    logger.info(f"turbo-filemanager {turbofilemanager_version}")

    logger.info(f"Start {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info(f"Kosuke Nakano, ({datetime.today().strftime('%Y')})")
    logger.info("E-mail: kousuke_1123@icloud.com")
    logger.info("")

    if args.job == "medit":
        if args.job == "medit":
            file_names = [os.path.join(machine_handler_env_dir, "machine_data.yaml")]
        else:
            raise NotImplementedError
            # file_names = os.path.join(machine_handler_env_dir, job_manager_env_dir)

        for file_name in file_names:
            call([args.editor, file_name])

    logger.info(f"job type = {args.job}")
    logger.debug(f"local machine = {args.local_machine}")
    logger.debug(f"client machine = {args.client_machine}")
    logger.debug(f"server machine = {args.server_machine}")
    logger.debug(f"rsync include_list = {args.include}")
    logger.debug(f"rysnc exclude_list = {args.exclude}")
    logger.debug(f"dryrun flag = {args.dryrun}")
    logger.debug(f"delele flag = {args.delete}")
    logger.debug(f"safe_mode flag = {args.safe_mode}")
    logger.info("")

    # cwd = os.getcwd()

    if args.delete:
        logger.warning("Warning!! you have turned on the rsync --delete option!")
        logger.warning("Warning!! This is a destructive rsync operation.")
        # logger.warning("Warning!! This option has not completely been tested.")
        # raise NotImplementedError

    if args.delete and args.safe_mode:
        logger.warning("Warning!! you have turned on the rsync --delete option!")
        logger.warning("Warning!! This is a destructive rsync operation.")
        logger.warning(
            "Warning!! You should switch off the safe_mode option, -safe_mode"
        )

    if args.job == "put":
        transfer = Data_transfer(
            local_machine_name=args.local_machine,
            client_machine_name=args.client_machine,
            server_machine_name=args.server_machine,
        )

        transfer.put_objects(
            from_objects=[],
            include_list=args.include,
            exclude_list=args.exclude,
            dryrun_flag=args.dryrun,
            delete_flag=args.delete,
        )

    elif args.job == "get":
        transfer = Data_transfer(
            local_machine_name=args.local_machine,
            client_machine_name=args.client_machine,
            server_machine_name=args.server_machine,
        )
        transfer.get_objects(
            from_objects=[],
            include_list=args.include,
            exclude_list=args.exclude,
            dryrun_flag=args.dryrun,
            delete_flag=args.delete,
        )

    else:
        logger.error(f"job = {args.job} is not implemented.")

    logger.info(
        f"End turbofile-manager {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
    )
