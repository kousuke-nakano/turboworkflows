#!/usr/bin/env python
# coding: utf-8

# python packages
import os
import shutil
import glob
import asyncio
import pathlib
import filecmp
import matplotlib.pyplot as plt
import numpy as np

# Logger
from logging import getLogger, StreamHandler, Formatter

# turbogenius package
from turbogenius.pyturbo.lrdmc import LRDMC
from turbogenius.pyturbo.utils.execute import run
from turbogenius.pyturbo.utils.utility import (
    pygrep_lineno,
    get_line_from_file,
)

# turboworkflow packages
from turboworkflows.workflow_encapsulated import Workflow
from turboworkflows.workflow_lrdmc import LRDMC_workflow

logger = getLogger("Turbo-Workflows").getChild(__name__)


class LRDMC_ext_workflow(Workflow):
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
        # lrdmc
        lrdmc_input_files=["fort.10", "pseudo.dat"],
        lrdmc_rerun=False,
        lrdmc_max_continuation=2,
        lrdmc_pkl_name="lrdmc_genius",
        lrdmc_target_error_bar=2.0e-5,  # Ha
        lrdmc_trial_steps=150,
        lrdmc_bin_block=10,
        lrdmc_warmupblocks=5,
        lrdmc_correcting_factor=10,
        lrdmc_trial_etry=0.0,
        lrdmc_alat_list=[-0.20, -0.30, -0.40],
        lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
        lrdmc_num_walkers=-1,  # default -1 -> num of MPI process.
        lrdmc_twist_average=False,
        lrdmc_kpoints=[],
        lrdmc_force_calc_flag=False,
        lrdmc_maxtime=172000,
        degree_poly=2,
    ):
        # job
        self.server_machine_name = server_machine_name
        self.cores = cores
        self.openmp = openmp
        self.queue = queue
        self.version = version
        self.sleep_time = sleep_time
        self.jobpkl_name = jobpkl_name
        # lrdmc
        self.lrdmc_input_files = lrdmc_input_files
        self.lrdmc_rerun = lrdmc_rerun
        self.lrdmc_max_continuation = lrdmc_max_continuation
        self.lrdmc_pkl_name = lrdmc_pkl_name
        self.lrdmc_target_error_bar = lrdmc_target_error_bar
        self.lrdmc_trial_steps = lrdmc_trial_steps
        self.lrdmc_bin_block = lrdmc_bin_block
        self.lrdmc_warmupblocks = lrdmc_warmupblocks
        self.lrdmc_num_walkers = lrdmc_num_walkers
        self.lrdmc_correcting_factor = lrdmc_correcting_factor
        self.lrdmc_nonlocalmoves = lrdmc_nonlocalmoves
        self.lrdmc_trial_etry = lrdmc_trial_etry
        self.lrdmc_alat_list = lrdmc_alat_list
        self.lrdmc_twist_average = lrdmc_twist_average
        self.lrdmc_kpoints = lrdmc_kpoints
        self.lrdmc_force_calc_flag = lrdmc_force_calc_flag
        self.lrdmc_maxtime = lrdmc_maxtime
        self.degree_poly = degree_poly
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
        os.chdir(self.root_dir)
        self.lrdmc_dir = os.path.join(self.root_dir)
        logger.info(f"Project root dir = {self.lrdmc_dir}")

        async def async_launch_(lrdmc_alat):
            os.chdir(self.lrdmc_dir)
            logger.info(f"alat={lrdmc_alat}")
            alat_dir = os.path.join(self.lrdmc_dir, f"alat_{lrdmc_alat}")
            if not os.path.isdir(os.path.join(alat_dir)):
                os.makedirs(alat_dir, exist_ok=True)
                logger.info(f"current dir = {os.getcwd()}")
                logger.info(f"input files = {self.lrdmc_input_files}")
                for file in self.lrdmc_input_files:
                    if os.path.isfile(file):
                        shutil.copy(
                            os.path.join(file),
                            os.path.join(alat_dir, os.path.basename(file)),
                        )
                    else:  # dir
                        if os.path.isdir(
                            os.path.join(alat_dir, os.path.basename(file))
                        ):
                            shutil.rmtree(
                                os.path.join(alat_dir, os.path.basename(file))
                            )
                        shutil.copytree(
                            os.path.join(file),
                            os.path.join(alat_dir, os.path.basename(file)),
                        )

            # check file/dir consistencies
            for file in self.lrdmc_input_files:
                if os.path.isfile(file):
                    assert filecmp.cmp(
                        os.path.join(self.lrdmc_dir, os.path.basename(file)),
                        os.path.join(alat_dir, os.path.basename(file)),
                        shallow=True,
                    )
                else:
                    dircmp = filecmp.dircmp(
                        os.path.join(self.lrdmc_dir, os.path.basename(file)),
                        os.path.join(alat_dir, os.path.basename(file)),
                        shallow=True,
                    )
                    assert len(dircmp.left_only) == 0
                    assert len(dircmp.right_only) == 0

            os.chdir(alat_dir)
            # await asyncio.sleep(60)
            lrdmc_workflow = LRDMC_workflow(
                # job
                server_machine_name=self.server_machine_name,
                cores=self.cores,
                openmp=self.openmp,
                queue=self.queue,
                version=self.version,
                sleep_time=self.sleep_time,
                jobpkl_name=self.jobpkl_name,
                # lrdmc
                lrdmc_rerun=self.lrdmc_rerun,
                lrdmc_max_continuation=self.lrdmc_max_continuation,
                lrdmc_pkl_name=self.lrdmc_pkl_name,
                lrdmc_target_error_bar=self.lrdmc_target_error_bar,
                lrdmc_trial_steps=self.lrdmc_trial_steps,
                lrdmc_bin_block=self.lrdmc_bin_block,
                lrdmc_warmupblocks=self.lrdmc_warmupblocks,
                lrdmc_num_walkers=self.lrdmc_num_walkers,
                lrdmc_correcting_factor=self.lrdmc_correcting_factor,
                lrdmc_nonlocalmoves=self.lrdmc_nonlocalmoves,
                lrdmc_trial_etry=self.lrdmc_trial_etry,
                lrdmc_alat=lrdmc_alat,
                lrdmc_twist_average=self.lrdmc_twist_average,
                lrdmc_kpoints=self.lrdmc_kpoints,
                lrdmc_force_calc_flag=self.lrdmc_force_calc_flag,
                lrdmc_maxtime=self.lrdmc_maxtime,
            )
            await lrdmc_workflow.async_launch()
            os.chdir(self.lrdmc_dir)

        async def async_gather_(lrdmc_alat_list):
            logger.info("=== Launch workflows ===")
            tsks = [
                asyncio.create_task(async_launch_(lrdmc_alat=lrdmc_alat))
                for lrdmc_alat in lrdmc_alat_list
            ]
            await asyncio.gather(*tsks)
            os.chdir(self.lrdmc_dir)

        # run LRDMC several alats
        os.chdir(self.lrdmc_dir)
        await async_gather_(lrdmc_alat_list=self.lrdmc_alat_list)
        os.chdir(self.lrdmc_dir)

        # run LRDMC extrapolation
        os.chdir(self.lrdmc_dir)
        logger.info(
            f"the polynomial degree for fitting of energies \
                with respect to alat^2 is {self.degree_poly}"
        )
        evsa_line = f"{self.degree_poly}  {len(self.lrdmc_alat_list)}  4  1\n"
        evsa_gnu_line = "# alat  energy  error\n"

        for alat in self.lrdmc_alat_list:
            dir_alat = os.path.join(self.lrdmc_dir, f"alat_{alat}")
            os.chdir(dir_alat)
            energy, error = LRDMC.read_energy(
                twist_average=self.lrdmc_twist_average
            )
            evsa_line = evsa_line + f"{np.abs(alat)} {energy} {error}\n"
            evsa_gnu_line = (
                evsa_gnu_line + f"{np.abs(alat)} {energy} {error}\n"
            )
            os.chdir(self.lrdmc_dir)
        os.chdir(self.lrdmc_dir)
        with open("evsa.in", "w") as f:
            f.writelines(evsa_line)
        with open("evsa.gnu", "w") as f:
            f.writelines(evsa_gnu_line)

        run(binary="funvsa.x", input_name="evsa.in", output_name="evsa.out")
        logger.info("Output from fitting process saved to evsa.out.")

        coeff_index = pygrep_lineno("evsa.out", "Coefficient found")
        coeff_list = []
        coeff_error_list = []
        for poly in range(self.degree_poly + 1):
            coeff_list.append(
                float(
                    get_line_from_file(
                        "evsa.out", coeff_index + 1 + poly
                    ).split()[1]
                )
            )
            coeff_error_list.append(
                float(
                    get_line_from_file(
                        "evsa.out", coeff_index + 1 + poly
                    ).split()[2]
                )
            )
        coeff_list.reverse()  # because const -> x -> x2 -> ... in evsa.out
        coeff_error_list.reverse()
        energy_ext = coeff_list[-1]
        error_ext = coeff_error_list[-1]
        logger.info(
            f"Extrapolated LRDMC energy (a->0) = \
                {energy_ext} Ha +- {error_ext} Ha"
        )
        self.output_values["energy"] = energy_ext
        self.output_values["error"] = error_ext
        poly = np.poly1d(coeff_list)

        os.chdir(self.lrdmc_dir)
        with open("evsa.gnu", "r") as f:
            lines = f.readlines()
            alat = []
            energy = []
            energy_error = []
            for line in lines:
                if "#" in line:
                    pass
                else:
                    alat.append(float(line.split()[0]))
                    energy.append(float(line.split()[1]))
                    energy_error.append(float(line.split()[2]))

        alat_squared = np.array(alat) ** 2
        alat_squared_extrapolated = np.linspace(
            0, np.max(alat_squared) * 1.10, 500
        )
        energy = np.array(energy)
        energy_error = np.array(energy_error)

        plt.figure()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.major.width"] = 1.0
        plt.rcParams["ytick.major.width"] = 1.0
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.linewidth"] = 1.5
        plt.xlim(
            [alat_squared_extrapolated.min(), alat_squared_extrapolated.max()]
        )
        plt.annotate(
            "E(alat->0) = {:.5f} Ha +- {:.5f} Ha".format(
                coeff_list[-1], coeff_error_list[-1]
            ),
            xy=(0.05, 0.05),
            xycoords="axes fraction",
        )
        plt.errorbar(
            alat_squared,
            energy,
            yerr=energy_error,
            color="black",
            marker="o",
            fmt="",
        )
        plt.plot(
            alat_squared_extrapolated,
            poly(alat_squared_extrapolated),
            color="blue",
            linestyle="dashed",
        )
        plt.xlabel(
            "alat$^2$ (Bohr$^2$)", fontname="Times New Roman", fontsize=14
        )
        plt.ylabel("Energy (Ha)", fontname="Times New Roman", fontsize=14)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(
            False
        )  # No offset for y-axis
        plt.gca().get_xaxis().get_major_formatter().set_useOffset(
            False
        )  # No offset for x-axis
        plt.savefig("Energy_vs_alat.png", bbox_inches="tight", pad_inches=0.2)
        plt.close()
        logger.info(
            "The graph of the extrapolation is saved as Energy_vs_alat.png"
        )

        # end
        logger.info("LRDMC-ext workflow ends.")
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
