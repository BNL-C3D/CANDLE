import os
import argparse
import yaml
from typing import List, Union, Type, TypeVar, Optional
from pathlib import Path
from pydantic import validator
from pydantic import BaseSettings as _BaseSettings
import radical.utils as ru
from radical.entk import AppManager
from ddmd_async import AsyncPipelineManager

_T = TypeVar("_T")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args

class BaseSettings(_BaseSettings):
    @classmethod
    def from_yaml(cls: Type[_T], filename: Union[str, Path]) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)

class CPUReqs(BaseSettings):
    processes: int = 1
    process_type: Optional[str]
    threads_per_process: int = 1
    thread_type: Optional[str]

    @validator("process_type")
    def process_type_check(cls, v):
        valid_process_types = {None, "MPI"}
        if v not in valid_process_types:
            raise ValueError(f"process_type must be one of {valid_process_types}")
        return v

    @validator("thread_type")
    def thread_type_check(cls, v):
        thread_process_types = {None, "OpenMP"}
        if v not in thread_process_types:
            raise ValueError(f"thread_type must be one of {thread_process_types}")
        return v

class GPUReqs(BaseSettings):
    processes: int = 0
    process_type: Optional[str]
    threads_per_process: int = 0
    thread_type: Optional[str]
    
    @validator("process_type")
    def process_type_check(cls, v):
        valid_process_types = {None, "MPI"}
        if v not in valid_process_types:
            raise ValueError(f"process_type must be one of {valid_process_types}")
        return v

    @validator("thread_type")
    def thread_type_check(cls, v):
        thread_process_types = {None, "OpenMP", "CUDA"}
        if v not in thread_process_types:
            raise ValueError(f"thread_type must be one of {thread_process_types}")
        return v

class BaseStageConfig(BaseSettings):
    pre_exec: List[str] = []
    executable: str = ""
    arguments: List[str] = []
    cpu_reqs: CPUReqs = CPUReqs()
    gpu_reqs: GPUReqs = GPUReqs()

class MolecularDynamicsStageConfig(BaseStageConfig):
    num_tasks: int = 1

class AggregationStageConfig(BaseStageConfig):
    num_tasks: int = 1

class MachineLearningStageConfig(BaseStageConfig):
    num_tasks: int = 1

class AgentStageConfig(BaseStageConfig):
    num_tasks: int = 1

class ExperimentConfig(BaseSettings):
    resource: str
    queue: str
    schema_: str
    project: str
    walltime_min: int
    cpus_per_node: int
    gpus_per_node: int
    num_nodes: int
    molecular_dynamics_stage: MolecularDynamicsStageConfig
    aggregation_stage: AggregationStageConfig
    machine_learning_stage: MachineLearningStageConfig
    agent_stage: AgentStageConfig


if __name__ == "__main__":

    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)

    appman = AppManager(
        hostname=os.environ["RMQ_HOSTNAME"],
        port=int(os.environ["RMQ_PORT"]),
        username=os.environ["RMQ_USERNAME"],
        password=os.environ["RMQ_PASSWORD"],
        autoterminate=False
    )

    appman.resource_desc = {
        "resource": cfg.resource,
        "queue": cfg.queue,
        "schema": cfg.schema_,
        "walltime": cfg.walltime_min,
        "project": cfg.project,
        "cpus": cfg.cpus_per_node * cfg.num_nodes,
        "gpus": cfg.gpus_per_node * cfg.num_nodes,
    }

    pipeline_manager = AsyncPipelineManager(cfg)


    do_separate = False
    if do_separate:
        # Simulation-only pipeline to produce data for upstream
        # (preproc, ML, agent) tasks
        sim_pipeline_init1 = pipeline_manager.generate_sim_pipeline()
        sim_pipeline_init2 = pipeline_manager.generate_sim_pipeline()
        appman.workflow = set([sim_pipeline_init1]+[sim_pipeline_init2])
        appman.run()

        # Two pipelines: one for preproc+ML+agent to run on first set
        # of simulations, and another to produce next set of simulations
        mlana_pipeline1 = pipeline_manager.generate_final_pipeline()
        mlana_pipeline2 = pipeline_manager.generate_final_pipeline()
        sim_pipeline = pipeline_manager.generate_sim_pipeline()
        appman.workflow = set([mlana_pipeline1] + [mlana_pipeline2] + [sim_pipeline])
        appman.run()

        # Last set of pipelines to execute. One is preproc+ML+agent
        # to run on previous simulation output; other comprises a
        # simulation stage followed by preproc+ML+agent for those
        # simulations.
        mlana_pipeline = pipeline_manager.generate_final_pipeline()
        complete_pipeline = pipeline_manager.generate_complete_pipeline()
        appman.workflow = set([mlana_pipeline] + [complete_pipeline])
        appman.run()

        # Finish with preproc+ML+agent, running on last set of
        # simulation data
        # final_pipeline = pipeline_manager.generate_final_pipeline()
        # appman.workflow = [final_pipeline]
        # appman.run()

    else:
        # This pipeline has the following stages:
        # (1) all simulation
        # (2) sim+preproc
        # (3) sim+ml
        # (4) sim+agent
        # (5) preproc
        # (6) ml
        # (7) agent
        # For testing, (2-4) execute twice; (5-7) operate on last of
        # simulation data produced in final iteration of (2-4)
        async_pipeline = pipeline_manager.generate_async_pipeline()
        appman.workflow = [async_pipeline]
        appman.run()

    appman.terminate()
