import numpy as np
from math import ceil
from typing import List
from radical.entk import Pipeline, Stage, Task

SUMMIT_CORES = 42
SUMMIT_GPU = 6
SF_SIM_TX = 1/40
SF_TX = 1/40
SF_TX_AGENT = 1/6
SF_NTASKS = 1/10
TX_SIM = 1360*SF_TX    # [s]
TX_PREPROC = 340*SF_TX    # [s]
TX_ML = 250*SF_TX     # [s]
TX_AGENT = 150*SF_TX  # [s]

NUM_SIM_TASKS = 960*SF_NTASKS  # GPU tasks
NUM_PREPROC_TASKS = 420*SF_NTASKS  # CPU tasks
NUM_ML_TASKS = 1 # GPU task
NUM_AGENT_TASKS_CPU = 6720 * SF_NTASKS # CPU task
NUM_AGENT_TASKS_GPU = 960 * SF_NTASKS # GPU task

# Three: sim+preproc, sim+ml, sim+agent
# N.B. Since Agent uses all hardware resources, may be better
# to instead async only first two (sim+preproc, sim+ml)
NUM_ASYNC_STAGES = 3

def generate_task(cfg, name, ttx) -> Task:
    task = Task()
    task.name = name
    # task.executable = "cat /dev/null; jsrun --bind rs -n{} -p{} -r6 -g1 -c1 ".format(cfg.cpu_reqs.dict()['processes'], cfg.cpu_reqs.dict()['processes']) + cfg.executable
    task.executable = cfg.executable
    task.arguments = ['%s' % ttx]
    task.pre_exec = cfg.pre_exec.copy()
    task.cpu_reqs = cfg.cpu_reqs.dict().copy()
    task.gpu_reqs = cfg.gpu_reqs.dict().copy()
    return task


def generate_ttx(nsamples, mu=20, stddev=10):
    normal = np.random.normal(mu, stddev, nsamples)
    return normal

# We will want to have a distribution of run-times

class AsyncPipelineManager:

    def __init__(self, cfg):
        self.cfg = cfg


    def generate_sim_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_stages(self.generate_sim_stage())
        return pipeline


    def generate_sim_stage(self) -> Stage:
        cfg = self.cfg.molecular_dynamics_stage
        stage = Stage()
        stage.name = "Simulation"

        # Generate normally-distributed pseudo-randoms for this
        # pipeline
        normal_rands = generate_ttx(cfg.num_tasks,
                                    TX_SIM, 0.005)

        # Number of simulation tasks per pipeline
        for t in range(0, cfg.num_tasks):
            tname = "sim." + str(t)
            task = generate_task(cfg, tname, normal_rands[t])
            stage.add_tasks(task)

        return stage


    def generate_sim_stage2(self) -> Stage:
        cfg = self.cfg.molecular_dynamics_stage
        stage = Stage()
        stage.name = "Simulation"

        # Generate normally-distributed pseudo-randoms for this
        # pipeline
        normal_rands = generate_ttx(cfg.num_tasks*2,
                                    TX_SIM, 0.005)

        # Number of simulation tasks per pipeline
        for t in range(0, int(cfg.num_tasks*2)):
            tname = "sim." + str(t)
            task = generate_task(cfg, tname, normal_rands[t])
            stage.add_tasks(task)

        return stage


    def generate_sp_stage2(self) -> Stage:
        cfg = self.cfg.molecular_dynamics_stage
        stage = Stage()
        stage.name = "SimPreproc"

        # Sim
        normal_rands = generate_ttx(cfg.num_tasks,
                                    TX_SIM, 0.005)
        for t in range(0, cfg.num_tasks):
            tname = "sim." + str(t)
            task = generate_task(cfg, tname, normal_rands[t])
            stage.add_tasks(task)

        # Preproc
        cfg = self.cfg.aggregation_stage
        normal_rands = generate_ttx(cfg.num_tasks, TX_PREPROC, 0.005)
        for t in range(0, cfg.num_tasks):
            tname = "preproc." + str(t)
            stage.add_tasks(generate_task(cfg, tname, normal_rands[t]))
        
        # ML
        cfg = self.cfg.machine_learning_stage
        normal_rands = generate_ttx(cfg.num_tasks, TX_ML, 0.005)
        for t in range(0, cfg.num_tasks):
            tname = "ml." + str(t)
            stage.add_tasks(generate_task(cfg, tname, normal_rands[t]))

        return stage


    def generate_spm_stage(self) -> Stage:
        stage = Stage()
        stage.name = "SimPreprocML"

        # Sim
        cfg = self.cfg.molecular_dynamics_stage
        normal_rands = generate_ttx(cfg.num_tasks,
                                    TX_SIM, 0.005)
        for t in range(0, cfg.num_tasks):
            tname = "sim." + str(t)
            task = generate_task(cfg, tname, normal_rands[t])
            stage.add_tasks(task)

        # Preproc
        cfg = self.cfg.aggregation_stage
        normal_rands = generate_ttx(cfg.num_tasks*2, TX_PREPROC, 0.005)
        for t in range(0, cfg.num_tasks*2):
            tname = "preproc." + str(t)
            stage.add_tasks(generate_task(cfg, tname, normal_rands[t]))
        
        # ML
        cfg = self.cfg.machine_learning_stage
        normal_rands = generate_ttx(cfg.num_tasks*2, TX_ML, 0.005)
        for t in range(0, cfg.num_tasks*2):
            tname = "ml." + str(t)
            stage.add_tasks(generate_task(cfg, tname, normal_rands[t]))

        return stage


    def generate_preproc_stage(self) -> Stage:
        cfg = self.cfg.aggregation_stage
        stage = Stage()
        stage.name = "Preprocessing"

        # Generate normally-distributed pseudo-randoms
        normal_rands = generate_ttx(cfg.num_tasks, TX_PREPROC, 0.005)

        for t in range(0, cfg.num_tasks):
            tname = "preproc." + str(t)
            stage.add_tasks(generate_task(cfg, tname, normal_rands[t]))

        return stage


    def generate_mlana_stage(self) -> Stage:
        cfg = self.cfg.machine_learning_stage
        stage = Stage()
        stage.name = "MachineLearning"

        # Generate normally-distributed pseudo-randoms
        normal_rands = generate_ttx(cfg.num_tasks, TX_ML, 0.005)

        for t in range(0, cfg.num_tasks):
            tname = "ml." + str(t)
            stage.add_tasks(generate_task(cfg, tname, normal_rands[t]))

        return stage


    def generate_mlana_stage2(self) -> Stage:
        cfg = self.cfg.machine_learning_stage
        stage = Stage()
        stage.name = "MachineLearning"

        # Generate normally-distributed pseudo-randoms
        normal_rands = generate_ttx(cfg.num_tasks*2, TX_ML, 0.005)

        for t in range(0, int(cfg.num_tasks*2)):
            tname = "ml." + str(t)
            stage.add_tasks(generate_task(cfg, tname, normal_rands[t]))

        return stage


    def generate_agent_stage(self) -> Stage:
        cfg = self.cfg.agent_stage
        stage = Stage()
        stage.name = "Agent"

        normal_rands = generate_ttx(cfg.num_tasks, TX_AGENT, 0.005)

        for t in range(0, cfg.num_tasks):
            tname = "agent." + str(t)
            task = generate_task(cfg, tname, normal_rands[t])
            stage.add_tasks(task)

        return stage


    def generate_agent_stage2(self) -> Stage:
        cfg = self.cfg.agent_stage
        stage = Stage()
        stage.name = "Agent"

        normal_rands = generate_ttx(cfg.num_tasks*2, TX_AGENT, 0.005)

        for t in range(0, int(cfg.num_tasks*2)):
            tname = "agent." + str(t)
            task = generate_task(cfg, tname, normal_rands[t])
            stage.add_tasks(task)

        return stage


    def generate_agent_stage3(self) -> Stage:
        cfg = self.cfg.agent_stage
        stage = Stage()
        stage.name = "Agent"

        normal_rands = generate_ttx(cfg.num_tasks*3, TX_AGENT, 0.005)

        for t in range(0, int(cfg.num_tasks*3)):
            tname = "agent." + str(t)
            task = generate_task(cfg, tname, normal_rands[t])
            stage.add_tasks(task)

        return stage


    def generate_complete_pipeline(self) -> Pipeline:
        pipeline = Pipeline()
        pipeline.add_stages(self.generate_sim_stage())
        pipeline.add_stages(self.generate_preproc_stage())
        pipeline.add_stages(self.generate_mlana_stage())
        pipeline.add_stages(self.generate_agent_stage())
        return pipeline


    def gen_async_stage(self, sname,
                        cfg1, tname1, tx1, stdev1,
                        cfg2, tname2, tx2, stdev2) -> Stage:
        stage = Stage()

        # Simulation Tasks
        stage.name = sname

        # Generate normally-distributed pseudo-randoms
        normal_rands = generate_ttx(cfg1.num_tasks,
                                    tx1, stdev1)

        # Number of simulation tasks per pipeline
        # Here, we divide the # tasks by the # of async stages
        # (of which there are three)
        for t in range(0, cfg1.num_tasks):
            tname = tname1 + '.' + str(t)
            stage.add_tasks(generate_task(cfg1, tname1, normal_rands[t]))
        
        # Generate normally-distributed pseudo-randoms
        normal_rands = generate_ttx(cfg2.num_tasks,
                                    tx2, stdev2)

        for t in range(0, cfg2.num_tasks):
            tname = tname2 + '.' + str(t)
            stage.add_tasks(generate_task(cfg2, tname2, normal_rands[t]))

        return stage


    def generate_async_pipeline(self) -> List[Pipeline]:
        pipeline = Pipeline()

        cfg_sim = self.cfg.molecular_dynamics_stage
        tname_sim = 'sim'
        stdev_sim = 0.005
        cfg_preproc = self.cfg.aggregation_stage
        tname_preproc = 'preproc'
        stdev_preproc = 0.005
        cfg_ml = self.cfg.machine_learning_stage
        tname_ml = 'ml'
        stdev_ml = 0.005
        cfg_agent = self.cfg.agent_stage
        tname_agent = 'agent'
        stdev_agent = 0.005

        """
        """
        pipeline.add_stages(self.generate_sim_stage2())
        pipeline.add_stages(self.generate_spm_stage())
        pipeline.add_stages(self.gen_async_stage(
            'AsyncSimPreproc1',
            cfg_preproc, tname_preproc, TX_PREPROC, stdev_preproc,
            cfg_ml, tname_ml, TX_ML, stdev_ml))
        pipeline.add_stages(self.generate_agent_stage3())
        """
        """

        return pipeline


    def generate_final_pipeline(self) -> List[Pipeline]:
        pipeline = Pipeline()
        pipeline.add_stages(self.generate_preproc_stage())
        pipeline.add_stages(self.generate_mlana_stage())
        pipeline.add_stages(self.generate_agent_stage())
        return pipeline
