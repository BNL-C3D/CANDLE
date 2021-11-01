import radical.utils as ru
from radical.entk import AppManager
from async_ddmd import AsyncPipelineManager


if __name__ == "__main__":

    args = parse_args()
    cfg = reading_config(args.config)

    appman = AppManager(
        hostname=os.environ["RMQ_HOSTNAME"],
        port=int(os.environ["RMQ_PORT"]),
        username=os.environ["RMQ_USERNAME"],
        password=os.environ["RMQ_PASSWORD"],
    )

    appman.resource_desc = {
        "resource": cfg.resource,
        "queue": cfg.queue,
        "schema": cfg.schema_,
        "walltime": cfg.walltime_min,
        "project": cfg.project,
        "cpus": cfg.cpus,
        "gpus": cfg.gpus,
    }

    pipeline_manager = AsyncPipelineManager(cfg)
    pipelines = pipeline_manager.generate_pipelines()
    appman.workflow = pipelines
    appman.run()
