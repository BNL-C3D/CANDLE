from radical.entk import Pipeline, Stage, Task


def generate_task(cfg) -> Task:
        task = Task()
        task.cpu_reqs = cfg.cpu_reqs
        task.gpu_reqs = cfg.gpu_reqs
        task.pre_exec = cfg.pre_exec
        task.executable = cfg.executable
        task.arguments = cfg.arguments
        return task

class AsyncPipelineManager:

    def __init__(self, cfg):
        self.cfg = cfg

    def generate_pipelines(self) -> List[Pipeline]:
        pipelines = [
            self.generate_molecular_dynamics_pipeline(),
            self.generate_aggregation_pipeline(),
            self.generate_machine_learning_pipeline(),
            self.generate_agent_pipeline(),
        ]
        return pipelines

    def generate_molecular_dynamics_pipeline(self) -> Pipeline:
        stage = Stage()
        stage.name = "MolecularDynamics"
        cfg = self.cfg.molecular_dynamics_stage

        for task_idx in range(cfg.num_sim_tasks):
            cfg.task_config.pdb_file = next(filenames)
            task = generate_task(cfg)
            stage.add_tasks(task)

        pipeline = Pipeline()
        pipeline.name = "MolecularDynamics"
        pipeline.add_stages(stage)

        return pipeline

    def generate_aggregation_pipeline(self) -> Pipeline:
        stage = Stage()
        stage.name = "Aggregation"
        cfg = self.cfg.aggregation_stage

        for task_idx in range(cfg.num_aggr_tasks)
            cfg.sim_task_output = collect_sim_results()
            task = generate_task(cfg)
            stage.add_tasks(task)

        pipeline = Pipeline()
        pipeline.name = "Aggregation"
        pipeline.add_stages(stage)

        return pipeline

    def generate_machine_learning_pipeline(self) -> Pipeline:
        stage = Stage()
        stage.name = "MachineLearning"
        cfg = self.cfg.machine_learning_stage

        for task_idx in range(cfg.num_ml_tasks)
            cfg.aggr_task_output = collect_aggr_results()
            task = generate_task(cfg)
            stage.add_tasks(task)

        pipeline = Pipeline()
        pipeline.name = "MachineLearning"
        pipeline.add_stages(stage)

        return pipeline

    def generate_agent_pipeline(self) -> Pipeline:
        stage = Stage()
        stage.name = "Agent"
        cfg = self.cfg.agent_stage

        cfg.ml_task_output = collect_ml_results()
        task = generate_task(cfg)
        stage.add_tasks(task)

        pipeline = Pipeline()
        pipeline.name = "Agent"
        pipeline.add_stages(stage)

        return pipeline
