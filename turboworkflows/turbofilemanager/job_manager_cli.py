# -*- coding: utf-8 -*-

import os
import re
import argparse
import shutil
from datetime import datetime
import pickle
import pathlib
import glob
import yaml

# define logger
from logging import getLogger, StreamHandler, Formatter

# import file-manager modules
from .file_manager_env import (
    file_manager_config_template_dir,
    file_manager_config_dir,
)

# from .job_manager import Job_submission

logger = getLogger("Turbo-Workflows").getChild(__name__)

try:
    from .._version import (
        version as turbofilemanager_version,
    )
except (ModuleNotFoundError, ImportError):
    turbofilemanager_version = "unknown"


class Monitor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.job_list_conter = 0
        self.job_dir_list = []
        self.job_pkl_list = []
        self.genius_pkl_list = []
        self.jobid_same_dir_dict = {}

    def show_detail(self, id):
        id = int(id)
        with open(os.path.join(self.job_pkl_list[id]), "rb") as f:
            job_handler = pickle.load(f)

        logger.info("--------------------------------------------------------------")
        logger.info(f"Detail of the jobID = {id}")
        logger.info("--------------------------------------------------------------")

        logger.info("==Local info.==")
        logger.info(
            f" - localhost dir = {os.path.dirname(os.path.join(self.job_pkl_list[id]))}"
        )
        logger.info("")

        logger.info("==Server info.==")
        logger.info(f" - server_machine_name = {job_handler.server_machine.name}")
        logger.info(f" - server dir = {job_handler.job_dir}")
        logger.info("")

        logger.info("==Job status info.==")
        logger.info(f" - job_number = {job_handler.job_number}")
        logger.info(f" - job_running = {job_handler.job_running}")
        logger.info(f" - job_submit_date = {job_handler.job_submit_date}")
        logger.info(f" - job_check_last_time = {job_handler.job_check_last_time}")
        logger.info(f" - job_fetch_date = {job_handler.job_fetch_date}")
        logger.info("")

        logger.info("==Job info.==")
        logger.info(f" - package = {job_handler.package}")
        logger.info(f" - binary-path = {job_handler.binary_path}")
        logger.info(f" - binary-name = {job_handler.binary}")
        logger.info(f" - jobname = {job_handler.jobname}")
        logger.info(f" - input_file = {job_handler.input_file}")
        logger.info(f" - output_file = {job_handler.output_file}")

    def show_tree(self):
        self.job_list_conter = 0
        self.job_dir_list = []
        logger.info("--------------------------------------------------------------")
        logger.info("TurboWorkflows job tree")
        logger.info("--------------------------------------------------------------")
        self.tree(path=self.root_dir)

    def tree(self, path, layer=0, is_last=False, indent_current="　"):
        if not pathlib.Path(path).is_absolute():
            path = str(pathlib.Path(path).resolve())

        def is_job_path(path):
            return glob.glob(os.path.join(path, "job_manager*.pkl"))

        current = os.path.basename(path)

        if not is_job_path(path):
            if layer == 0:
                logger.info("<" + current + "> <--- current dir")
            else:
                job_pkl_list = glob.glob(f"{path}/**/job_manager*.pkl", recursive=True)
                job_pkl_list.sort()
                if len(job_pkl_list) != 0:
                    branch = "└" if is_last else "├"
                    logger.info(
                        "{indent}{branch}<{dirname}>".format(
                            indent=indent_current,
                            branch=branch,
                            dirname=current,
                        )
                    )
        else:
            jobid_same_dir_list = []
            job_pkl_list = glob.glob(os.path.join(path, "job_manager*.pkl"))
            job_pkl_list.sort()
            for ii, job_manager_pkl_file in enumerate(job_pkl_list):
                is_last_item = True if ii == len(job_pkl_list) - 1 else False
                match = re.search(
                    r"job_manager_(\d+)\.pkl",
                    os.path.basename(job_manager_pkl_file),
                )
                if match:
                    genius_index = int(match.group(1))
                    file_pattern = os.path.join(
                        path, f"*_genius_{genius_index}.pkl"
                    )
                else:
                    file_pattern = os.path.join(path, "*_genius.pkl")

                genius_pkl_file_list = glob.glob(file_pattern)
                if not genius_pkl_file_list:
                    logger.debug(f'file not found: {file_pattern}')
                    genius_pkl_file = None
                else:
                    genius_pkl_file = genius_pkl_file_list[0]

                with open(job_manager_pkl_file, "rb") as f:
                    job_handler = pickle.load(f)
                    server_machine_name = job_handler.server_machine.name
                    job_number = job_handler.job_number
                    job_running = job_handler.job_running

                if genius_pkl_file is None:
                    job_comment = "is in queue"
                elif job_running:
                    job_comment = "is running"
                else:
                    job_comment = "is done"

                if layer == 0 and ii == 0:
                    indent = ""
                    branch = ""
                else:
                    indent = " " * (len(current) + 2) if layer == 0 else indent_current
                    branch = "└" if (is_last or layer == 0) and is_last_item else "├"

                logger.info(
                    "{indent}{branch}<{dirname}>-{job_number}({genius_pkl_file}) {job_comment} on {server_machine_name} (JOB-ID:{job_index})".format(
                        indent=indent,
                        branch=branch,
                        dirname=current,
                        job_comment=job_comment,
                        server_machine_name=server_machine_name,
                        job_number=job_number,
                        job_index=self.job_list_conter,
                        genius_pkl_file=os.path.basename(genius_pkl_file) if genius_pkl_file else None,
                    )
                )

                jobid_same_dir_list.append(self.job_list_conter)
                self.job_pkl_list.append(job_manager_pkl_file)
                self.genius_pkl_list.append(genius_pkl_file)
                self.job_dir_list.append(path)
                self.job_list_conter += 1

            self.jobid_same_dir_dict[path] = jobid_same_dir_list

        paths = sorted([
            p for p in glob.glob(path + "/*") if os.path.isdir(p) and glob.glob(f"{p}/**/job_manager*.pkl", recursive=True)
        ])
        logger.debug(f"paths={paths}")

        def is_last_path(i):
            return i == len(paths) - 1

        for i, p in enumerate(paths):
            indent_lower = indent_current
            if layer != 0:
                indent_lower += "　　" if is_last else "│　"

            if os.path.isdir(p):
                self.tree(
                    p,
                    layer=layer + 1,
                    is_last=is_last_path(i),
                    indent_current=indent_lower,
                )

def do_show(monitor, jobid):
    monitor.show_tree()
    if jobid != -1:
        monitor.show_detail(id=jobid)

def do_check(monitor, jobid):
    monitor.show_tree()

    pkls = []
    if jobid != -1:
        pkls = [monitor.job_pkl_list[jobid]]
    else:
        pkls = glob.glob("job_manager*.pkl")

    for pkl in pkls:
        with open(pkl, "rb") as f:
            submission = pickle.load(f)
            status = submission.jobcheck()

            if status:
                logger.info(f"JobNumber {submission.job_number} is still running on {submission.server_machine.name}.")
            else:
                logger.info(f"JobNumber {submission.job_number} is done. Plz. fetch from {submission.server_machine.name}.")

def do_del(monitor, jobid, dry_run=True):
    monitor.show_tree()

    if jobid == -1:
        logger.info("Please specify a Job-ID you want to remove by -id JOB-ID")
        return

    if dry_run:
        logger.info("[dry_run]")

    for key, item in monitor.jobid_same_dir_dict.items():
        if jobid in item:
            target_dir = key
            # jobid_same_dir_list = item
            break
    else:
        logger.error(f"JOB-ID {jobid} not found.")
        return

    # no continuation genius jobs
    if os.path.isfile(os.path.join(target_dir, "job_manager.pkl")):

        logger.info(f'remove {os.path.join(target_dir, "job_manager.pkl")}')
        if not dry_run:
            os.remove(os.path.join(target_dir, "job_manager.pkl"))

        file_pattern = os.path.join(target_dir, "*_genius.pkl")
        genius_pkl_file_list = glob.glob(file_pattern)

        for genius_pkl_file in genius_pkl_file_list:
            logger.info(f'remove {genius_pkl_file}')
            if not dry_run:
                os.remove(genius_pkl_file)

        if os.path.isdir(os.path.join(target_dir, "pkl")):
            logger.info(f'remove directory {os.path.join(target_dir, "pkl")}')
            if not dry_run:
                shutil.rmtree(os.path.join(target_dir, "pkl"))

    # continuation genius jobs
    else:
        if (
            os.path.basename(monitor.job_pkl_list[jobid])
            == "job_manager_0.pkl"
        ):
            logger.info("Remove all jobs!")

            removed_job_pkl_file_list = glob.glob(
                os.path.join(target_dir, "job_manager*.pkl")
            )

            removed_genius_pkl_file_list = glob.glob(
                os.path.join(target_dir, "*_genius_*.pkl")
            )

            for removed_job_pkl_file in removed_job_pkl_file_list:
                logger.info(f'remove {removed_job_pkl_file}')
                if not dry_run:
                    os.remove(removed_job_pkl_file)

            for removed_genius_pkl_file in removed_genius_pkl_file_list:
                logger.info(f'remove {removed_genius_pkl_file}')
                if not dry_run:
                    os.remove(removed_genius_pkl_file)

            if os.path.isdir(os.path.join(target_dir, "pkl")):
                logger.info(f'remove directory {os.path.join(target_dir, "pkl")}')
                if not dry_run:
                    shutil.rmtree(os.path.join(target_dir, "pkl"))

        else:
            match = re.search(
                r"job_manager_(\d+)\.pkl",
                os.path.basename(monitor.job_pkl_list[jobid]),
            )
            if match:
                removed_genius_index = int(match.group(1))
                logger.info(f"removed_genius_index={removed_genius_index}")
                new_latest_genius_index = removed_genius_index - 1
            else:
                logger.error("not found: job_manager_(\d+)\.pkl")
                return
                #raise ValueError

            removed_job_pkl_file_list = []
            removed_genius_pkl_file_list = []

            job_pkl_file_list = glob.glob(
                os.path.join(target_dir, "job_manager*.pkl")
            )
            for job_pkl_file in job_pkl_file_list:
                match = re.search(
                    r"job_manager_(\d+)\.pkl",
                    os.path.basename(job_pkl_file),
                )
                if match:
                    genius_index = int(match.group(1))
                    if genius_index >= removed_genius_index:
                        removed_job_pkl_file_list.append(job_pkl_file)

            genius_pkl_file_list = glob.glob(
                os.path.join(target_dir, "*_genius_*.pkl")
            )
            for genius_pkl_file in genius_pkl_file_list:
                match = re.search(
                    r"(.*)_genius_(\d+)\.pkl",
                    os.path.basename(genius_pkl_file),
                )
                if match:
                    genius_prefix = str(match.group(1))
                    genius_index = int(match.group(2))
                    if genius_index >= removed_genius_index:
                        removed_genius_pkl_file_list.append(genius_pkl_file)

            for removed_job_pkl_file in removed_job_pkl_file_list:
                logger.info(f'remove {removed_job_pkl_file}')
                if not dry_run:
                    os.remove(removed_job_pkl_file)

            for removed_genius_pkl_file in removed_genius_pkl_file_list:
                logger.info(f'remove {removed_genius_pkl_file}')
                if not dry_run:
                    os.remove(removed_genius_pkl_file)

            if os.path.isdir(os.path.join(target_dir, "pkl")):

                for removed_genius_pkl_file in removed_genius_pkl_file_list:
                    pkl = os.path.join(target_dir, "pkl", os.path.basename(removed_genius_pkl_file))
                    if os.path.isfile(pkl):
                        logger.info(f'remove {pkl}')
                        if not dry_run:
                            os.remove(pkl)

                pklsrc = os.path.join(target_dir, "pkl", f"{genius_prefix}_genius_{new_latest_genius_index}.pkl")
                pkldst = os.path.join(target_dir, "pkl", f"{genius_prefix}_genius_latest.pkl")
                if os.path.isfile(pklsrc):
                    logger.info(f'copy {pklsrc} to {pkldst}')
                    if not dry_run:
                        shutil.copy(pklsrc, pkldst)

    logger.warning(f"Deleted JOB-ID = {jobid}!!")

def job_manager_cli():
    root_dir = os.getcwd()

    job_list = ["show", "del", "check"]

    # check if machine info file exists
    machine_info_yaml = os.path.join(file_manager_config_dir, "machine_data.yaml")
    try:
        with open(machine_info_yaml, "r") as yf:
            machine_list = yaml.safe_load(yf).keys()
    except FileNotFoundError:
        print(f"The yaml file={machine_info_yaml} is not found!!")
        # check config dir exists.
        if not os.path.isdir(file_manager_config_dir):
            print(
                f"{file_manager_config_dir} is not found. Probably, this is the first run."
            )
            #os.makedirs(file_manager_config_dir, exist_ok=True)
            shutil.copytree(file_manager_config_template_dir, file_manager_config_dir)
            print(f"{file_manager_config_dir} has been generated.")
            print(f"plz. edit {machine_info_yaml}")
            return
        else:
            raise FileNotFoundError

    # define the parser
    parser = argparse.ArgumentParser(
        epilog=f"turbo-jobmanager {turbofilemanager_version}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Job type:
    parser.add_argument(
        "job", help=f"Choose job type from {job_list}", choices=job_list
    )
    # Job ID:
    parser.add_argument("-id", "--jobid", help="Specify jobid", default=-1, type=int)

    # qstat machine
    parser.add_argument(
        "-s",
        "--server_machine",
        help="server machine for qstat or qdel",
        type=str,
        choices=machine_list,
        default="localhost",
    )
    # logger
    parser.add_argument(
        "-log", "--log_level", choices=["DEBUG", "INFO"], default="INFO"
    )
    # dry_run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry-run. does not actually delete pkl files.",
    )

    # parse the input values
    args = parser.parse_args()
    # parsed_parameter_dict = vars(args)

    logger = getLogger("Turbo-Workflows")
    logger.setLevel(args.log_level)
    stream_handler = StreamHandler()
    stream_handler.setLevel(args.log_level)
    if args.log_level == "DEBUG":
        handler_format = Formatter("%(name)s l-%(lineno)d %(message)s")
    else:
        handler_format = Formatter("%(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    logger.info("--------------------------------------------------------------")
    logger.info(f"turbo-jobmanager {turbofilemanager_version}")
    logger.info(f"Start {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Kosuke Nakano, NIMS ({datetime.today().strftime('%Y')})")
    logger.info("E-mail: kousuke_1123@icloud.com")
    # logger.info("--------------------------------------------------------------")

    monitor = Monitor(root_dir=root_dir)

    if args.job == "show":
        do_show(monitor, args.jobid)

    elif args.job == "check":
        do_check(monitor, args.jobid)

    elif args.job == "del":
        do_del(monitor, args.jobid, dry_run=args.dry_run)

    logger.info("--------------------------------------------------------------")
    logger.info(
        f"End turbo-jobmanager {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info("--------------------------------------------------------------")


def main():
    pass


if __name__ == "__main__":
    main()
