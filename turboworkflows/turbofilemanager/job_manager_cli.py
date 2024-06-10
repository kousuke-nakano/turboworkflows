# -*- coding: utf-8 -*-

import os
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
    machine_handler_env_dir,
    file_manager_config_dir,
    job_manager_env_dir,
    machine_handler_env_template_dir,
    job_manager_env_template_dir,
)
from .machine_handler import Machine
from .job_manager import Job_submission

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
        self.job_pkl_list = []
        self.job_dir_list = []

    def show_dir_jobdir(self, jobid):
        logger.info(
            f"jobid({jobid}) = {pathlib.Path(self.job_dir_list[jobid]).relative_to(os.getcwd())}"
        )
        # umm it does not work

    def show_detail(self, id):
        id = int(id)
        with open(os.path.join(self.job_pkl_list[id]), "rb") as f:
            job_handler = pickle.load(f)

        logger.info(f"--Detail of the jobID = {id}--")

        logger.info("==Local info.==")
        logger.info(
            f" - local dir = {os.path.dirname(os.path.join(self.job_pkl_list[id]))}"
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
        logger.info(f" - job_status = {job_handler.job_status}")
        logger.info("")

        logger.info("==Job info.==")
        logger.info(f" - package = {job_handler.package}")
        logger.info(
            f" - binary = {os.path.join(job_handler.binary_path, job_handler.binary)}"
        )
        logger.info(f" - cores = {job_handler.cores}")
        logger.info(f" - openmp = {job_handler.openmp}")
        logger.info(f" - queue = {job_handler.queue}")
        logger.info(f" - cpns = {job_handler.cpns}")
        logger.info(f" - mpi_per_node = {job_handler.mpi_per_node}")
        logger.info(f" - max_job_run = {job_handler.max_job_run}")
        logger.info(f" - max_job_submit = {job_handler.max_job_submit}")
        logger.info(f" - max_time = {job_handler.max_time}")
        logger.info(f" - jobname = {job_handler.jobname}")
        logger.info(f" - input_file = {job_handler.input_file}")
        logger.info(f" - output_file = {job_handler.output_file}")

    def show_tree(self, show_files=False):
        self.job_list_conter = 0
        self.job_dir_list = []
        self.tree(path=self.root_dir, show_files=show_files)

    def tree(
        self,
        path,
        layer=0,
        is_last=False,
        indent_current="　",
        show_files=False,
    ):
        if not pathlib.Path(path).is_absolute():
            path = str(pathlib.Path(path).resolve())

        def is_job_path(path):
            return os.path.isfile(os.path.join(path, "job_manager.pkl"))

        current = os.path.basename(path)
        if layer == 0:
            if not is_job_path(path):
                logger.info("<" + current + "> <--- current dir")
            else:
                with open(os.path.join(path, "job_manager.pkl"), "rb") as f:
                    job_handler = pickle.load(f)
                    server_machine_name = job_handler.server_machine.name
                    job_number = job_handler.job_number
                    job_running = job_handler.job_running
                    if job_running:
                        job_comment = "is running"
                    else:
                        job_comment = "is done"
                    logger.info(
                        "<{dirname}, current dir>-{job_number} {job_comment} on {server_machine_name} (id:{job_index})".format(
                            dirname=current,
                            job_comment=job_comment,
                            server_machine_name=server_machine_name,
                            job_number=job_number,
                            job_index=self.job_list_conter,
                        )
                    )

                    self.job_list_conter += 1
                    self.job_dir_list.append(path)
                    self.job_pkl_list.append(os.path.join(path, "job_manager.pkl"))

        else:
            branch = "└" if is_last else "├"
            if not is_job_path(path):
                job_files = glob.glob(f"{path}/**/job_manager.pkl", recursive=True)
                if len(job_files) != 0:
                    logger.info(
                        "{indent}{branch}<{dirname}>".format(
                            indent=indent_current,
                            branch=branch,
                            dirname=current,
                        )
                    )
            else:
                with open(os.path.join(path, "job_manager.pkl"), "rb") as f:
                    job_handler = pickle.load(f)
                    server_machine_name = job_handler.server_machine.name
                    job_number = job_handler.job_number
                    job_running = job_handler.job_running
                    if job_running:
                        job_comment = "is running"
                    else:
                        job_comment = "is done"
                    logger.info(
                        "{indent}{branch}<{dirname}>-{job_number} {job_comment} on {server_machine_name} (id:{job_index})".format(
                            indent=indent_current,
                            branch=branch,
                            dirname=current,
                            job_comment=job_comment,
                            server_machine_name=server_machine_name,
                            job_number=job_number,
                            job_index=self.job_list_conter,
                        )
                    )

                    self.job_list_conter += 1
                    self.job_dir_list.append(path)
                    self.job_pkl_list.append(os.path.join(path, "job_manager.pkl"))

        paths = sorted(
            [p for p in glob.glob(path + "/*") if os.path.isdir(p) or os.path.isfile(p)]
        )
        if show_files:
            paths = [
                p
                for p in paths
                if glob.glob(f"{p}/**/job_manager.pkl", recursive=True)
                or os.path.isfile(p)
            ]
        else:
            paths = [
                p for p in paths if glob.glob(f"{p}/**/job_manager.pkl", recursive=True)
            ]

        def is_last_path(i):
            return i == len(paths) - 1

        for i, p in enumerate(paths):

            indent_lower = indent_current
            if layer != 0:
                indent_lower += "　　" if is_last else "│　"

            if os.path.isfile(p):
                if show_files:
                    branch = "└" if is_last_path(i) else "├"
                    logger.info(
                        "{indent}{branch}{filename}".format(
                            indent=indent_lower,
                            branch=branch,
                            filename=p.split("/")[::-1][0],
                        )
                    )
            if os.path.isdir(p):
                self.tree(
                    p,
                    layer=layer + 1,
                    is_last=is_last_path(i),
                    indent_current=indent_lower,
                )


def job_manager_cli():
    root_dir = os.getcwd()

    job_list = ["toss", "fetch", "show", "dir", "del", "check", "stat"]

    # check if machine info file exists
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

            if not os.path.isdir(job_manager_env_dir):
                print(f"{job_manager_env_dir} is not found.")
                os.makedirs(file_manager_config_dir, exist_ok=True)
                shutil.copytree(job_manager_env_template_dir, job_manager_env_dir)
                print(f"{job_manager_env_dir} has been generated.")
                print(f"plz. also edit directories and files in {job_manager_env_dir}")

            return
        else:
            raise FileNotFoundError

    # check if job info file exists
    if not os.path.isdir(job_manager_env_dir):
        print(f"{job_manager_env_dir} is not found.")
        os.makedirs(file_manager_config_dir, exist_ok=True)
        shutil.copytree(job_manager_env_template_dir, job_manager_env_dir)
        print(f"{job_manager_env_dir} has been generated.")
        print(f"plz. edit directories and files in {job_manager_env_dir}")
        return

    # define the parser
    parser = argparse.ArgumentParser(
        epilog=f"turbo-jobmanager {turbofilemanager_version}",
        usage="see [https://github.com/kousuke-nakano/turbofilemanager]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Job type:
    parser.add_argument(
        "job", help=f"Choose job type from {job_list}", choices=job_list
    )
    # Job ID:
    parser.add_argument("-id", "--jobid", help="Specify jobid", default=-1)
    # files
    parser.add_argument(
        "-f", "--files", help="show files", action="store_true", default=False
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

    # jobsubmission (for toss)
    parser.add_argument("-p", "--package", help="package")
    parser.add_argument("-ver", "--version", help="version", default=None)
    parser.add_argument("-b", "--binary", help="binary", default=None)
    parser.add_argument("-core", "--cores", help="core", default=1, type=int)
    parser.add_argument("-omp", "--openmp", help="openmp", default=1, type=int)
    parser.add_argument("-q", "--queue", help="queue", default=None)
    parser.add_argument("-j", "--jobname", help="jobname", default="job")
    parser.add_argument("-i", "--inputfile", help="input file", default=None)
    parser.add_argument("-pre", "--preoption", help="preoption", default=[], nargs="*")
    parser.add_argument(
        "-post", "--postoption", help="postoption", default=[], nargs="*"
    )
    parser.add_argument(
        "-nompi", "--nompi", help="nompi", action="store_true", default=False
    )
    parser.add_argument("-o", "--outputfile", help="output file", default="out.o")
    parser.add_argument(
        "-inc",
        "--include",
        help="specify an resync include list",
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "-exc",
        "--exclude",
        help="specify an resync exclude list (default is read from exclude_list.txt in the current dir.)",
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

    # parse the input values
    args = parser.parse_args()
    # parsed_parameter_dict = vars(args)

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
    logger.info(f"turbo-jobmanager {turbofilemanager_version}")
    logger.info(f"Start {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info(f"Kosuke Nakano, ({datetime.today().strftime('%Y')})")
    logger.info("E-mail: kousuke_1123@icloud.com")
    logger.info("")

    if args.job == "toss":
        if len(args.preoption):
            preoption = " ".join(args.preoption)
        else:
            preoption = None
        if len(args.postoption):
            postoption = " ".join(args.postoption)
        else:
            postoption = None
        submission = Job_submission(
            local_machine_name=args.local_machine,
            client_machine_name=args.client_machine,
            server_machine_name=args.server_machine,
            package=args.package,
            version=args.version,
            binary=args.binary,
            cores=args.cores,
            openmp=args.openmp,
            queue=args.queue,
            jobname=args.jobname,
            input_file=args.inputfile,
            preoption=preoption,
            postoption=postoption,
            nompi=args.nompi,
            output_file=args.outputfile,
            pkl_name="job_manager.pkl",
        )
        submission.generate_script()

        if len(args.include) > 0:
            args.include.append("submit.sh")
            args.include.append(args.inputfile)

        # job submission
        job_flag = submission.job_submit(
            from_objects=[],
            include_list=args.include,
            exclude_list=args.exclude,
            dryrun_flag=args.dryrun,
            delete_flag=args.delete,
        )

        if job_flag:
            logger.error("job submission is successful")
        else:
            logger.error("job submission is failure")

    if args.job == "fetch":
        with open("job_manager.pkl", mode="rb") as f:
            submission = pickle.load(f)
        logger.info(f"Fetching from {submission.server_machine.name}.")
        submission.fetch_job(
            from_objects=[],
            include_list=args.include,
            exclude_list=args.exclude,
            dryrun_flag=args.dryrun,
            delete_flag=args.delete,
        )

    elif args.job == "show":
        if args.jobid == -1:
            monitor = Monitor(root_dir=root_dir)
            monitor.show_tree(show_files=args.files)
            with open(os.path.join(root_dir, ".jobmonitor.tmp"), "wb") as f:
                pickle.dump(monitor, f)
        else:
            try:
                with open(os.path.join(root_dir, ".jobmonitor.tmp"), "rb") as f:
                    monitor = pickle.load(f)
            except FileNotFoundError:
                monitor = Monitor(root_dir=root_dir)
                monitor.show_tree(show_files=args.files)
                with open(os.path.join(root_dir, ".jobmonitor.tmp"), "wb") as f:
                    pickle.dump(monitor, f)

            monitor.show_detail(id=args.jobid)

            # monitor.chdir_jobdir(jobid=19)

    elif args.job == "stat":
        if args.server_machine is None:
            logger.warning("Choose a machine for qstat by '-m' or '--machine'")
            logger.warning(list(machine_list))
        else:
            machine = Machine(args.server_machine)
            logger.warning(f"Joblist on machine {machine.name}")
            job_list, _ = machine.get_job_list()
            i = 1
            for uu, job in enumerate(job_list.split("\n")):
                if machine.username in job:
                    logger.info("  ".join(job.split()))
                    i += 1
    else:
        try:
            with open(os.path.join(root_dir, ".jobmonitor.tmp"), "rb") as f:
                monitor = pickle.load(f)
        except FileNotFoundError:
            monitor = Monitor(root_dir=root_dir)
            monitor.show_tree(show_files=args.files)
            with open(os.path.join(root_dir, ".jobmonitor.tmp"), "wb") as f:
                pickle.dump(monitor, f)

        if args.job == "dir":
            if args.jobid == -1:
                logger.error("Specify the job ID where you want to go.")
            monitor.show_dir_jobdir(jobid=args.jobid)

        elif args.job == "del":
            if args.jobid != -1 and args.server_machine is not None:
                machine = Machine(args.server_machine)
                logger.info(f"delite job = {args.jobid} on {machine.name}.")
                machine.delete_job(jobid=args.jobid)
            else:
                if args.jobid != -1:
                    with open(
                        os.path.join(monitor.job_pkl_list[args.jobid]), "rb"
                    ) as f:
                        submission = pickle.load(f)
                else:
                    with open("job_manager.pkl", mode="rb") as f:
                        submission = pickle.load(f)
                logger.info(
                    f"delite job = {submission.job_number} on {submission.server_machine.name}."
                )
                submission.delete_job()

        elif args.job == "check":
            if args.jobid != -1:
                with open(os.path.join(monitor.job_pkl_list[args.jobid]), "rb") as f:
                    submission = pickle.load(f)
            else:
                with open("job_manager.pkl", mode="rb") as f:
                    submission = pickle.load(f)

            if submission.jobcheck():
                logger.info(
                    f"JobNumber {submission.job_number} is still running on {submission.server_machine.name}."
                )
            else:
                logger.info(
                    f"JobNumber {submission.job_number} is done. Plz. fetch from {submission.server_machine.name}."
                )

    logger.info(
        f"End turbo-jobmanager {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def main():
    pass


if __name__ == "__main__":
    main()
