from typing import List
from radical.entk import Pipeline, Stage, Task


def generate_task(cfg) -> Task:
    """ Generate a task based on configuration.
    """
    # Create a new task object
    task = Task()

    # Set task properties, pulled from configuration file
    task.executable = cfg.executable
    task.arguments = cfg.arguments.copy()
    task.pre_exec = cfg.pre_exec.copy()
    task.cpu_reqs = cfg.cpu_reqs.dict().copy()
    task.gpu_reqs = cfg.gpu_reqs.dict().copy()

    return task

class PipelineManager:
    """ Manages stages and tasks for a given pipeline.
    """


    def __init__(self, cfg):
        self.cfg = cfg


    def generate_pipeline(self) -> List[Pipeline]:
        """ Generate a single pipeline comprising all stages.
        """
        # Create a list of stage objects
        stages = []

        # Create simulation stage
        s1 = Stage()
        s1.name = "MolecularDynamics"
        cfg = self.cfg.molecular_dynamics_stage
        for task_idx in range(cfg.num_tasks):
            task = generate_task(cfg)
            stage.add_tasks(task)
        stages.append(s1)

        # Create aggregator stage
        s2 = Stage()
        s2.name = "Aggregation"
        cfg = self.cfg.aggregation_stage
        for task_idx in range(cfg.num_tasks):
            task = generate_task(cfg)
            stage.add_tasks(task)
        stages.append(s2)

        # Create simulation stage
        s3 = Stage()
        s3.name = "MachineLearning"
        cfg = self.cfg.machine_learning_stage
        for task_idx in range(cfg.num_tasks):
            task = generate_task(cfg)
            stage.add_tasks(task)
        stages.append(s3)

        # Create simulation stage
        s4 = Stage()
        s4.name = "Agent"
        cfg = self.cfg.agent_stage
        for task_idx in range(cfg.num_tasks):
            task = generate_task(cfg)
            stage.add_tasks(task)
        stages.append(s4)

        # Create the pipeline object
        pipeline = Pipeline()
        pipeline.name = "Pipeline"

        # Add stage to the pipeline
        pipeline.add_stages(stages)

        return pipeline


    def generate_pipelines(self) -> List[Pipeline]:
        """ Generate a list of pipelines.
        """
        pipelines = [
            self.generate_molecular_dynamics_pipeline(),
            self.generate_aggregation_pipeline(),
            self.generate_machine_learning_pipeline(),
            self.generate_agent_pipeline(),
        ]
        return pipelines


    def generate_molecular_dynamics_pipeline(self) -> Pipeline:
        """ Create a pipeline for simulations.
        """
        # Create a stage object
        stage = Stage()
        stage.name = "MolecularDynamics"

        # Handle to simulation task configuration
        cfg = self.cfg.molecular_dynamics_stage

        # Create a number of tasks, defined in the configuration file, for this
        # stage, and add each task to the stage
        for task_idx in range(cfg.num_tasks):
            task = generate_task(cfg)
            stage.add_tasks(task)

        # Create the pipeline object
        pipeline = Pipeline()
        pipeline.name = "MolecularDynamics"

        # Add stage
        pipeline.add_stages(stage)

        return pipeline


    def generate_aggregation_pipeline(self) -> Pipeline:
        """ Create a pipeline for data aggregation.
        """

        # Create a stage object
        stage = Stage()
        stage.name = "Aggregation"

        # Handle to aggregation task configuration
        cfg = self.cfg.aggregation_stage

        # Create a number of tasks, defined in the configuration file, for this
        # stage, and add each task to the stage
        for task_idx in range(cfg.num_tasks):
            task = generate_task(cfg)
            stage.add_tasks(task)

        # Create the pipeline object
        pipeline = Pipeline()
        pipeline.name = "Aggregation"

        # Add stage to the pipeline
        pipeline.add_stages(stage)

        return pipeline


    def generate_machine_learning_pipeline(self) -> Pipeline:
        """ Create a pipeline for machine learning.
        """
        # Create the stage object
        stage = Stage()
        stage.name = "MachineLearning"

        # Handle to the machine learning task configuration
        cfg = self.cfg.machine_learning_stage

        # Create a number of tasks, defined in the configuration file, for this
        # stage, and add each task to the stage
        for task_idx in range(cfg.num_tasks):
            task = generate_task(cfg)
            stage.add_tasks(task)

        # Create the pipeline object
        pipeline = Pipeline()
        pipeline.name = "MachineLearning"

        # Add stage to the pipeline
        pipeline.add_stages(stage)

        return pipeline


    def generate_agent_pipeline(self) -> Pipeline:
        """ Create a pipeline for inference.
        """
        # Create the stage object
        stage = Stage()
        stage.name = "Agent"

        # Handle to agent task configuration
        cfg = self.cfg.agent_stage

        # Create a number of tasks, defined in the configuration file, for this
        # stage, and add each task to the stage
        for task_idx in range(cfg.num_tasks):
            task = generate_task(cfg)
            stage.add_tasks(task)

        # Create the pipeline object
        pipeline = Pipeline()
        pipeline.name = "Agent"

        # Add stage to the pipeline
        pipeline.add_stages(stage)

        return pipeline
