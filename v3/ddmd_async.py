from typing import List
from radical.entk import Pipeline, Stage, Task
#from main import BaseStageConfig


#def generate_task(cfg: BaseStageConfig) -> Task:
def generate_task(cfg) -> Task:
    task = Task()
    task.executable = cfg.executable
    task.arguments = cfg.arguments.copy()
    task.pre_exec = cfg.pre_exec.copy()
    task.cpu_reqs = cfg.cpu_reqs.dict().copy()
    task.gpu_reqs = cfg.gpu_reqs.dict().copy()
    return task

class AsyncPipelineManager:

    def __init__(self, cfg):
        self.cfg = cfg

    def generate_pipeline_stages(self):
        self.pipeline.add_stages(self.generate_molecular_dynamics_stage())
        self.pipeline.add_stages(self.generate_aggregation_stage())
        self.pipeline.add_stages(self.generate_machine_learning_stage())
        ##self.pipeline.add_stages(self.generate_agent_stage())

    def generate_pipeline_stages2(self):
        self.pipeline.add_stages(self.generate_molecular_dynamics_stage())
        self.pipeline.add_stages(self.generate_aggregation_stage())
        self.pipeline.add_stages(self.generate_machine_learning_stage())
        self.pipeline.add_stages(self.generate_agent_stage())

    def generate_pipelines(self) -> List[Pipeline]:
        pipelines = []
        for _ in range(self.cfg.num_nodes):
            self.pipeline = Pipeline()
            self.generate_pipeline_stages()
            pipelines.append(self.pipeline)
        sleep(25)
        for _ in range(self.cfg.num_nodes):
            self.pipeline = Pipeline()
            self.generate_pipeline_stages2()
            pipelines.append(self.pipeline)
        return pipelines

    def generate_molecular_dynamics_stage(self) -> Stage:
        stage = Stage()
        stage.name = "MolecularDynamics"
        cfg = self.cfg.molecular_dynamics_stage

        for task_idx in range(cfg.num_tasks):
            #cfg.task_config.pdb_file = next(filenames)
            task = generate_task(cfg)
            stage.add_tasks(task)

        return stage

    def generate_aggregation_stage(self) -> Stage:
        stage = Stage()
        stage.name = "Aggregation"
        cfg = self.cfg.aggregation_stage

        for task_idx in range(cfg.num_tasks):
            #cfg.sim_task_output = collect_sim_results()
            task = generate_task(cfg)
            stage.add_tasks(task)

        return stage

    def generate_machine_learning_stage(self) -> Stage:
        stage = Stage()
        stage.name = "MachineLearning"
        cfg = self.cfg.machine_learning_stage

        for task_idx in range(cfg.num_tasks):
            #cfg.aggr_task_output = collect_aggr_results()
            task = generate_task(cfg)
            stage.add_tasks(task)

        return stage

    def generate_agent_stage(self) -> Stage:
        stage = Stage()
        stage.name = "Agent"
        cfg = self.cfg.agent_stage

        #cfg.ml_task_output = collect_ml_results()
        task = generate_task(cfg)
        stage.add_tasks(task)

        return stage
