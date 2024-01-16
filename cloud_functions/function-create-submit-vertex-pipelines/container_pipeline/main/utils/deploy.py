from datetime import datetime
from pathlib import Path
from typing import Union
from google.cloud import aiplatform
from .scheduler_helpers import list_delete_jobs, create_scheduler_job


def deploy_pipeline(
    compiled_pipeline: str,
    enable_caching: bool,
    params_dict: dict,
    service_account: str,
    project_id: str,
    pipeline_root: str,
    pipeline_labels: dict,
    network: str,
    schedule: Union[None, str] = None,
    time_zone: str = "America/Santiago",
    region: str = "us-central1",
) -> None:

    # Parse name from compiled pipeline file
    pipeline_name = Path(compiled_pipeline).stem.replace("_", "-")
    current_date = datetime.utcnow().strftime("%Y%m%d-%H%M")

    if not schedule:
        aiplatform.init(project=project_id, staging_bucket=pipeline_root)
        # Starts Pipeline job from json compiled file
        pipeline_job = aiplatform.PipelineJob(
            display_name=pipeline_name,
            pipeline_root=pipeline_root,
            template_path=compiled_pipeline,
            job_id=f"{pipeline_name}-{current_date}",
            enable_caching=enable_caching,
            parameter_values=params_dict,
            labels=pipeline_labels,
            failure_policy="fast",
        )

        pipeline_job.submit(service_account=service_account, network=network)
        # pipeline_job.run(service_account=service_account, network=network)

    else:
        # Removes existing scheduled jobs
        list_delete_jobs(name=pipeline_name, project_id=project_id, region=region)
        schedule = schedule.replace('-', ' ')

        payload = {}
        with open(compiled_pipeline, "r") as f:
            payload["pipeline_spec"] = f.read()

        parameters = {
            "pipeline_name": pipeline_name,
            "service_account": service_account,
            "params_dict": params_dict,
            "labels": pipeline_labels,
            "network": network,
            "cached": enable_caching,
        }

        payload["parameters"] = parameters

        create_scheduler_job(
            name=pipeline_name,
            payload=payload,
            project_id=project_id,
            service_account=service_account,
            schedule=schedule,
            time_zone=time_zone,
            region=region,
        )
