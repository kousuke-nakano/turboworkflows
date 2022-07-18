#!/usr/bin/env python
# coding: utf-8

# python packages
import numpy as np
import os, sys
import shutil
import pickle
import time
import asyncio
import pathlib

#Logger
from logging import config, getLogger, StreamHandler, Formatter, FileHandler
logger = getLogger('Turbo-Workflows').getChild(__name__)

# turboworkflows packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_turboworkflows.turboworkflows_env import turbo_workflows_root
from workflow_encapsulated import eWorkflow, Workflow

class Variable:
    def __init__(self,
                 label='xxxx',
                 vtype='file',  # file or value
                 name='xxxx',  # filename or value's label
                 ):
        self.label=label
        self.vtype=vtype
        self.name=name

class Launcher:
    def __init__(self,
                 cworkflows_list=[] # cWorkflow()
                 ):
        self.cworkflows_list = cworkflows_list
        self.cworkflows_dict = {cworkflows.label:cworkflows for cworkflows in cworkflows_list}
        self.dependency_dict = self.__solve_dependency()
        self.topological_orders = self.__get_topological_orders()
        self.launcher_root_dir = os.getcwd()

    def launch(self):
        for topological_order in self.topological_orders:
            for label in topological_order:
                os.chdir(self.launcher_root_dir)
                cworkflows = self.cworkflows_dict[label]
                self.solve_Variables(cworkflows)
                cworkflows.launch()
                os.chdir(self.launcher_root_dir)

    def async_launch(self):
        async def async_gather_(topological_orders):
            async def async_launch_(topological_order):
                for label in topological_order:
                    os.chdir(self.launcher_root_dir)
                    cworkflows = self.cworkflows_dict[label]
                    self.solve_Variables(cworkflows)
                    await cworkflows.async_launch()
                    os.chdir(self.launcher_root_dir)
                os.chdir(self.launcher_root_dir)

            # run the topological order list, independently
            tsks = [asyncio.create_task(async_launch_(topological_order)) for topological_order in topological_orders]
            await asyncio.gather(*tsks)
        asyncio.run(async_gather_(topological_orders=self.topological_orders))

    def __get_value(self, v):
        label = v.label
        vtype = v.vtype
        name = v.name
        if vtype == "file":
            assert name in getattr(self.cworkflows_dict[label], 'output_files')
            dirname = getattr(self.cworkflows_dict[label], 'dirname')
            filepath = os.path.join(dirname, name)
            p=pathlib.Path(filepath)
            return p.resolve().relative_to(p.cwd())
        else:
            rvalue = getattr(self.cworkflows_dict[label], 'output_values')
            return rvalue[name]

    def __solve_dependency(self):
        dependency_dict={}
        for cworkflows in self.cworkflows_list:
            dep_labels_list=[]
            for key, value in cworkflows.__dict__.items():
                if isinstance(value, Workflow):
                    for key, vv in value.__dict__.items():
                        if isinstance(vv, list):
                            if any([isinstance(v, Variable) for v in vv]):
                                for v in vv:
                                    if isinstance(v, Variable):
                                        dep_labels_list.append(v.label)
                        else:
                            if isinstance(vv, Variable):
                                dep_labels_list.append(vv.label)

                # find Variables in the cworkflow attributes (list)
                elif isinstance(value, list):
                    if any([isinstance(v, Variable) for v in value]):
                        for v in value:
                            if isinstance(v, Variable):
                                dep_labels_list.append(v.label)

                # find Variables in the cworkflow attributes (str, int, float ...)
                else:
                    if isinstance(value, Variable):
                        dep_labels_list.append(value.label)

            dependency_dict[cworkflows.label]=tuple(set(dep_labels_list))

        return dependency_dict

    def __get_topological_orders(self):
        from paradag import DAG, dag_run

        # divide the groups to independent groups
        dag = DAG()
        for key in self.dependency_dict.keys():
            dag.add_vertex(key)
        for key, dependency_list in self.dependency_dict.items():
            for dependency in dependency_list:
                dag.add_edge(dependency, key)

        def get_successors_set(a):
            successors_set=[]
            for successors in dag.successors(a):
                successors_set+=get_successors_set(successors)
            successors_set += [a]
            return set(successors_set)

        independent_groups_list=[get_successors_set(start) for start in dag.all_starts()]

        # set topological orders
        topological_orders_list=[]
        for label_list in independent_groups_list:
            dag = DAG()
            for label in label_list:
                dag.add_vertex(label)
            for label in label_list:
                dependency_list=self.dependency_dict[label]
                for dependency in dependency_list:
                    dag.add_edge(dependency, label)

            topological_orders_list.append(dag_run(dag))

        return topological_orders_list

    def solve_Variables(self, cworkflows):
        for key, value in cworkflows.__dict__.items():
            # find Variables in the workflow attributes in the cworkflow
            if isinstance(value, Workflow):
                for key, vv in value.__dict__.items():
                    if isinstance(vv, list):
                        if any([isinstance(v, Variable) for v in vv]):
                            solved_v_list=[]
                            for v in vv:
                                if isinstance(v, Variable):
                                    solved_v=self.__get_value(v)
                                else:
                                    solved_v=v
                                solved_v_list.append(solved_v)
                            setattr(value, key, solved_v_list)
                    else:
                        if isinstance(vv, Variable):
                            solved_v=self.__get_value(vv)
                            setattr(value, key, solved_v)

            # find Variables in the cworkflow attributes (list)
            elif isinstance(value, list):
                if any([isinstance(v, Variable) for v in value]):
                    solved_v_list = []
                    for v in value:
                        if isinstance(v, Variable):
                            solved_v = self.__get_value(v)
                        else:
                            solved_v = v
                        solved_v_list.append(solved_v)
                    setattr(cworkflows, key, solved_v_list)

            # find Variables in the cworkflow attributes (str, int, float ...)
            else:
                if isinstance(value, Variable):
                    solved_v=self.__get_value(value)
                    setattr(cworkflows, key, solved_v)

if __name__ == "__main__":
    logger = getLogger("Turbo-Workflows")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    from workflow_lrdmc import LRDMC_workflow
    from workflow_vmc import VMC_workflow

    os.chdir(os.path.join(turbo_workflows_root, "tests", "launchers"))
    cworkflows_list=[]
    """
    for i in range(4):
        clrdmc_workflow = cWorkflow(
                         label=f'clrdmc-workflow-{i}',
                         dirname=f'clrdmc-workflow-{i}',
                         input_files=['fort.10', 'pseudo.dat'],
                         workflow=LRDMC_workflow(
                                    ## job
                                    server_machine_name="kagayaki",
                                    cores=64,
                                    openmp=1,
                                    queue="DEFAULT",
                                    version="stable",
                                    sleep_time=30, # sec.
                                    jobpkl_name="job_manager",
                                    ## lrdmc
                                    lrdmc_max_continuation=1,
                                    lrdmc_pkl_name="lrdmc_genius",
                                    lrdmc_target_error_bar=1.0e-3, # Ha
                                    lrdmc_trial_steps= 150,
                                    lrdmc_bin_block = 10,
                                    lrdmc_warmupblocks = 5,
                                    lrdmc_correcting_factor=10,
                                    lrdmc_trial_etry=-13.4,
                                    lrdmc_alat=-0.30,
                                    lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
                                    lrdmc_num_walkers = -1, # default -1 -> num of MPI process.
                                    lrdmc_twist_average=False,
                                    lrdmc_kpoints=[],
                                    lrdmc_force_calc_flag=False,
                                    lrdmc_maxtime=172000,
                                )
                        )

        clrdmc_workflow_n = cWorkflow(
                         label=f'clrdmc-workflow-n-{i}',
                         dirname=f'clrdmc-workflow-n-{i}',
                         input_files=[Variable(label=f'clrdmc-workflow-{i}', vtype='file', name='fort.10'),
                                      Variable(label=f'clrdmc-workflow-{i}', vtype='file', name='pseudo.dat')],
                         workflow=LRDMC_workflow(
                                    ## job
                                    server_machine_name="kagayaki",
                                    cores=64,
                                    openmp=1,
                                    queue="DEFAULT",
                                    version="stable",
                                    sleep_time=30, # sec.
                                    jobpkl_name="job_manager",
                                    ## lrdmc
                                    lrdmc_max_continuation=1,
                                    lrdmc_pkl_name="lrdmc_genius",
                                    lrdmc_target_error_bar=1.0e-3, # Ha
                                    lrdmc_trial_steps= 150,
                                    lrdmc_bin_block = 10,
                                    lrdmc_warmupblocks = 5,
                                    lrdmc_correcting_factor=5,
                                    lrdmc_trial_etry=Variable(label=f'clrdmc-workflow-{i}', vtype='value', name='energy'),
                                    lrdmc_alat=-0.30,
                                    lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
                                    lrdmc_num_walkers = -1, # default -1 -> num of MPI process.
                                    lrdmc_twist_average=False,
                                    lrdmc_kpoints=[],
                                    lrdmc_force_calc_flag=False,
                                    lrdmc_maxtime=172000,
                                )
                        )
        cworkflows_list.append(clrdmc_workflow)
        cworkflows_list.append(clrdmc_workflow_n)
        clrdmc_workflow.output_files = ["fort.10", 'pseudo.dat']
        clrdmc_workflow.output_values = {'energy': -27.5}
    """

    cvmc_workflow = eWorkflow(
        label=f'cvmc-workflow',
        dirname=f'cvmc-workflow',
        input_files=['fort.10', 'pseudo.dat'],
        workflow=VMC_workflow(
            ## job
            server_machine_name="kagayaki",
            cores=64,
            openmp=1,
            queue="DEFAULT",
            version="stable",
            sleep_time=30, # sec.
            jobpkl_name="job_manager",
            ## vmc
            vmc_max_continuation=1,
            vmc_pkl_name="vmc_genius",
            vmc_target_error_bar=1.0e-3, # Ha
            vmc_trial_steps= 150,
            vmc_bin_block = 10,
            vmc_warmupblocks = 5,
            vmc_num_walkers = -1, # default -1 -> num of MPI process.
            vmc_twist_average=False,
            vmc_kpoints=[],
            vmc_force_calc_flag=False,
            vmc_maxtime=172000,
        )
    )

    clrdmc_workflow = eWorkflow(
        label=f'clrdmc-workflow',
        dirname=f'clrdmc-workflow',
        input_files=[Variable(label=f'cvmc-workflow', vtype='file', name='fort.10'),
                     Variable(label=f'cvmc-workflow', vtype='file', name='pseudo.dat')],
        workflow=LRDMC_workflow(
            ## job
            server_machine_name="kagayaki",
            cores=64,
            openmp=1,
            queue="DEFAULT",
            version="stable",
            sleep_time=30,  # sec.
            jobpkl_name="job_manager",
            ## lrdmc
            lrdmc_max_continuation=1,
            lrdmc_pkl_name="lrdmc_genius",
            lrdmc_target_error_bar=1.0e-3,  # Ha
            lrdmc_trial_steps=150,
            lrdmc_bin_block=10,
            lrdmc_warmupblocks=5,
            lrdmc_correcting_factor=10,
            lrdmc_trial_etry=Variable(label=f'cvmc-workflow', vtype='value', name='energy'),
            lrdmc_alat=-0.30,
            lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
            lrdmc_num_walkers=-1,  # default -1 -> num of MPI process.
            lrdmc_twist_average=False,
            lrdmc_kpoints=[],
            lrdmc_force_calc_flag=False,
            lrdmc_maxtime=172000,
        )
    )

    cworkflows_list=[cvmc_workflow, clrdmc_workflow]
    launcher=Launcher(cworkflows_list=cworkflows_list)
    launcher.launch()
    #launcher.async_launch()



