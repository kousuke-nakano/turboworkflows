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
                 cworkflows_list=[], # cWorkflow() list
                 turbo_workflows_log_level="INFO",
                 turbo_genius_log_level="INFO",
                 pyturbo_log_level="INFO",
                 file_manager_log_level="INFO",
                 log_name="turboworkflows.log",
                 dependency_graph_draw=False,
                 ):
        
        # set loggers!!
        logger_w = getLogger("Turbo-Workflows")
        logger_w.setLevel(turbo_workflows_log_level)
        stream_handler_w = StreamHandler()
        stream_handler_w.setLevel(turbo_workflows_log_level)
        handler_format_w = Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
        stream_handler_w.setFormatter(handler_format_w)
        file_handler_w = FileHandler(log_name, 'a')
        file_handler_w.setLevel(turbo_workflows_log_level)
        file_handler_w.setFormatter(handler_format_w)
        logger_w.addHandler(file_handler_w)
        logger_w.addHandler(stream_handler_w)
        
        logger_t = getLogger('Turbo-Genius')
        logger_t.setLevel(turbo_genius_log_level)
        stream_handler_t = StreamHandler()
        stream_handler_t.setLevel(turbo_genius_log_level)
        handler_format_t = Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
        stream_handler_t.setFormatter(handler_format_t)
        file_handler_t = FileHandler(log_name, 'a')
        file_handler_t.setLevel(turbo_genius_log_level)
        file_handler_t.setFormatter(handler_format_t)
        logger_t.addHandler(file_handler_t)
        logger_t.addHandler(stream_handler_t)
        
        logger_p = getLogger('pyturbo')
        logger_p.setLevel(pyturbo_log_level)
        stream_handler_p = StreamHandler()
        stream_handler_p.setLevel(pyturbo_log_level)
        handler_format_p = Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
        stream_handler_p.setFormatter(handler_format_p)
        file_handler_p = FileHandler(log_name, 'a')
        file_handler_p.setLevel(pyturbo_log_level)
        file_handler_p.setFormatter(handler_format_p)
        logger_p.addHandler(file_handler_p)
        logger_p.addHandler(stream_handler_p)
        
        logger_f = getLogger('file-manager')
        logger_f.setLevel(file_manager_log_level)
        stream_handler_f = StreamHandler()
        stream_handler_f.setLevel(file_manager_log_level)
        handler_format_f = Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
        stream_handler_f.setFormatter(handler_format_f)
        file_handler_f = FileHandler(log_name, 'a')
        file_handler_f.setLevel(file_manager_log_level)
        file_handler_f.setFormatter(handler_format_f)
        logger_f.addHandler(file_handler_f)
        logger_f.addHandler(stream_handler_f)

        # attributes
        self.cworkflows_list = cworkflows_list
        self.cworkflows_dict = {cworkflows.label:cworkflows for cworkflows in cworkflows_list}
        self.dependency_dict = self.__solve_dependency(dependency_graph_draw=dependency_graph_draw)
        self.topological_orders = self.__get_topological_orders()
        self.launcher_root_dir = os.getcwd()

    def launch(self):
        os.chdir(self.launcher_root_dir)
        asyncio.run(self.async_launch())

    async def async_launch(self):
        os.chdir(self.launcher_root_dir)
        async def async_gather_(topological_orders):
            for label in topological_orders:
                os.chdir(self.launcher_root_dir)
                assert isinstance(label, list)
                cworkflows_list=[self.cworkflows_dict[l] for l in label]
                [self.solve_Variables(cworkflows) for cworkflows in cworkflows_list]
                tsks = [asyncio.create_task(cworkflows.async_launch()) for cworkflows in cworkflows_list]
                await asyncio.gather(*tsks)
                os.chdir(self.launcher_root_dir)
            os.chdir(self.launcher_root_dir)
        await async_gather_(topological_orders=self.topological_orders)
        #asyncio.run(async_gather_(topological_orders=self.topological_orders))
        os.chdir(self.launcher_root_dir)

        """ to be deleted
            async def async_launch_(topological_order):
                for label in topological_order:
                    os.chdir(self.launcher_root_dir)
                    if isinstance(label, str) or isinstance(label, int):
                        cworkflows = self.cworkflows_dict[label]
                        self.solve_Variables(cworkflows)
                        await cworkflows.async_launch()
                        os.chdir(self.launcher_root_dir)
                    elif isinstance(label, list):
                        cworkflows_list=[self.cworkflows_dict[l] for l in label]
                        [self.solve_Variables(cworkflows) for cworkflows in cworkflows_list]
                        tsks = [asyncio.create_task(cworkflows.async_launch()) for cworkflows in cworkflows_list]
                        await asyncio.gather(*tsks)
                        os.chdir(self.launcher_root_dir)
                    else:
                        logger.error(f"label={label} is not expected.")
                        raise ValueError
                    os.chdir(self.launcher_root_dir)
                os.chdir(self.launcher_root_dir)

            # run the topological order list, independently
            tsks = [asyncio.create_task(async_launch_(topological_order)) for topological_order in topological_orders]
            await asyncio.gather(*tsks)
            os.chdir(self.launcher_root_dir
            
        asyncio.run(async_gather_(topological_orders=self.topological_orders))
        """

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

    def __solve_dependency(self, dependency_graph_draw=False):
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

        if dependency_graph_draw:
            from graphviz import Digraph
            G = Digraph(format="png")
            G.attr("node", shape="squared")
            for key in dependency_dict.keys():
                G.node(key)
            for key, dependency_list in dependency_dict.items():
                for dependency in dependency_list:
                    G.edge(dependency, key)
            G.render("graphs")

        return dependency_dict

    def __get_topological_orders(self):
        from paradag import DAG, dag_run

        # divide the groups into independent groups
        dag = DAG()
        for key in self.dependency_dict.keys():
            dag.add_vertex(key)
        for key, dependency_list in self.dependency_dict.items():
            for dependency in dependency_list:
                dag.add_edge(dependency, key)

        topological_orders=dag_run(dag)
        logger.info(topological_orders)

        topological_orders_list_depth=[]
        depth_dict={}
        for label in topological_orders:
            # recursive function!, which returns the longest path from the root
            def get_predecessors_set(label, current_depth):
                predecessors_depth=[]
                if len(dag.predecessors(label)) == 0:
                    predecessors_depth.append(current_depth)
                else:
                    for predecessors in dag.predecessors(label):
                        predecessors_depth.append(get_predecessors_set(predecessors, current_depth+1))
                return np.max(predecessors_depth) # returns the longest root!
            max_depth=get_predecessors_set(label=label, current_depth=0)
            depth_dict[label]=max_depth

        # grouping the labels for each depth.
        logger.info(depth_dict)
        depth_set=set(depth_dict.values())
        for dep in depth_set:
            group=[label for label, depth in depth_dict.items() if depth==dep]
            topological_orders_list_depth.append(group)

        return topological_orders_list_depth

        """ to be deleted
        def get_successors_set(a):
            successors_set=[]
            for successors in dag.successors(a):
                successors_set+=get_successors_set(successors)
            successors_set += [a]
            return set(successors_set)

        independent_groups_list=[get_successors_set(start) for start in dag.all_starts()]
        logger.info(independent_groups_list)

        # set topological orders
        topological_orders_list=[]
        for label_list in independent_groups_list:
            dag_grouping = DAG()
            for label in label_list:
                dag_grouping.add_vertex(label)
            for label in label_list:
                dependency_list=self.dependency_dict[label]
                for dependency in dependency_list:
                    dag_grouping.add_edge(dependency, label)

            topological_orders_list.append(dag_run(dag_grouping))

        # here, regrouping topological_orders_list to run independent runs as parallelly as possible
        # Thanks Andrea, here, we seek the longest path from the root for each node
        topological_orders_list_mod=[]
        for topological_orders in topological_orders_list:
            topological_orders_list_depth=[]
            depth_dict={}
            for label in topological_orders:
                # recursive function!, which returns the longest path from the root
                def get_predecessors_set(label, current_depth):
                    predecessors_depth=[]
                    if len(dag.predecessors(label)) == 0:
                        predecessors_depth.append(current_depth)
                    else:
                        for predecessors in dag.predecessors(label):
                            predecessors_depth.append(get_predecessors_set(predecessors, current_depth+1))
                    return np.max(predecessors_depth) # returns the longest root!
                max_depth=get_predecessors_set(label=label, current_depth=0)
                depth_dict[label]=max_depth

            # grouping the labels for each depth.
            logger.info(depth_dict)
            depth_set=set(depth_dict.values())
            for dep in depth_set:
                group=[label for label, depth in depth_dict.items() if depth==dep]

                if len(group)==1:
                    topological_orders_list_depth.append(group[0])
                else:
                    topological_orders_list_depth.append(group)

            topological_orders_list_mod.append(topological_orders_list_depth)

        #return topological_orders_list
        logger.info(topological_orders_list_mod)
        return topological_orders_list_mod
        """

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
    from workflow_lrdmc import LRDMC_workflow
    from workflow_vmc import VMC_workflow

    os.chdir(os.path.join(turbo_workflows_root, "tests", "launchers_test"))
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

    cvmc_workflow_d = eWorkflow(
        label=f'cvmc-workflow-d',
        dirname=f'cvmc-workflow-d',
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

    cvmc_workflow_d = eWorkflow(
        label=f'cvmc-workflow-d',
        dirname=f'cvmc-workflow-d',
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

    cvmc_workflow_ind = eWorkflow(
        label=f'cvmc-workflow-ind',
        dirname=f'cvmc-workflow-ind',
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

    cworkflows_list = [cvmc_workflow,cvmc_workflow_d,cvmc_workflow_ind]

    for i in range(2):
        cworkflows_list.append(eWorkflow(
            label=f'clrdmc-workflow-{i}',
            dirname=f'clrdmc-workflow-{i}',
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
                lrdmc_alat=-i*0.1,
                lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
                lrdmc_num_walkers=-1,  # default -1 -> num of MPI process.
                lrdmc_twist_average=False,
                lrdmc_kpoints=[],
                lrdmc_force_calc_flag=False,
                lrdmc_maxtime=172000,
            )
        )
    )

    for i in range(2):
        cworkflows_list.append(eWorkflow(
            label=f'clrdmc-workflow-cc-{i}',
            dirname=f'clrdmc-workflow-cc-{i}',
            input_files=[Variable(label=f'clrdmc-workflow-{i}', vtype='file', name='fort.10'),
                         Variable(label=f'clrdmc-workflow-{i}', vtype='file', name='pseudo.dat')],
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
                lrdmc_alat=-i*0.1,
                lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
                lrdmc_num_walkers=-1,  # default -1 -> num of MPI process.
                lrdmc_twist_average=False,
                lrdmc_kpoints=[],
                lrdmc_force_calc_flag=False,
                lrdmc_maxtime=172000,
            )
        )
    )

    cworkflows_list.append(eWorkflow(
        label=f'clrdmc-workflow-ccc',
        dirname=f'clrdmc-workflow-ccc',
        input_files=[Variable(label=f'clrdmc-workflow-cc-0', vtype='file', name='fort.10'),
                     Variable(label=f'cvmc-workflow-d', vtype='file', name='pseudo.dat')],
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
            lrdmc_alat=- 0.3,
            lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
            lrdmc_num_walkers=-1,  # default -1 -> num of MPI process.
            lrdmc_twist_average=False,
            lrdmc_kpoints=[],
            lrdmc_force_calc_flag=False,
            lrdmc_maxtime=172000,
        )
    )
    )

    cworkflows_list.append(eWorkflow(
        label=f'clrdmc-workflow-ccc-a',
        dirname=f'clrdmc-workflow-ccc-a',
        input_files=[Variable(label=f'clrdmc-workflow-cc-0', vtype='file', name='fort.10'),
                     Variable(label=f'clrdmc-workflow-cc-0', vtype='file', name='pseudo.dat')],
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
            lrdmc_trial_etry=Variable(label=f'clrdmc-workflow-cc-0', vtype='value', name='energy'),
            lrdmc_alat=- 0.3,
            lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
            lrdmc_num_walkers=-1,  # default -1 -> num of MPI process.
            lrdmc_twist_average=False,
            lrdmc_kpoints=[],
            lrdmc_force_calc_flag=False,
            lrdmc_maxtime=172000,
        )
    )
    )

    cworkflows_list.append(eWorkflow(
        label=f'clrdmc-workflow-ind',
        dirname=f'clrdmc-workflow-ind',
        input_files=[Variable(label=f'cvmc-workflow-ind', vtype='file', name='fort.10'),
                     Variable(label=f'cvmc-workflow-ind', vtype='file', name='pseudo.dat')],
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
            lrdmc_trial_etry=Variable(label=f'cvmc-workflow-ind', vtype='value', name='energy'),
            lrdmc_alat=- 0.3,
            lrdmc_nonlocalmoves="dlatm",  # tmove, dla, dlatm
            lrdmc_num_walkers=-1,  # default -1 -> num of MPI process.
            lrdmc_twist_average=False,
            lrdmc_kpoints=[],
            lrdmc_force_calc_flag=False,
            lrdmc_maxtime=172000,
        )
    )
    )

    launcher=Launcher(cworkflows_list=cworkflows_list)
    #launcher.launch()
    launcher.async_launch()



