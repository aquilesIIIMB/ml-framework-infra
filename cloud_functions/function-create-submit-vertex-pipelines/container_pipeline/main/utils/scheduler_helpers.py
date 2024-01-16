import os
import re
import json
from typing import List
from google.cloud import scheduler


def list_jobs(name: str, project_id: str, region: str) -> List[str]:
    """
    Lists existing scheduled jobs that match with name
    """
    client = scheduler.CloudSchedulerClient()
    request = scheduler.ListJobsRequest(
        parent=f"projects/{project_id}/locations/{region}"
    )
    page_result = client.list_jobs(request=request)

    return [response.name for response in page_result if re.search('/' + name, response.name)]


def delete_job(name: str) -> None:
    """
    Deletes a scheduled job given its name
    """
    client = scheduler.CloudSchedulerClient()
    request = scheduler.DeleteJobRequest(name=name)
    client.delete_job(request=request)
    print(f"job: {name} DELETED")


def list_delete_jobs(name: str, project_id: str, region: str = "us-central1") -> None:
    """
    Searches for existing jobs, if a match is found, deletes the job
    """
    existing_jobs = list_jobs(name=name, project_id=project_id, region=region)
    if existing_jobs:
        for job in existing_jobs:
            delete_job(job)


def create_scheduler_job(
    name: str,
    payload: dict,
    project_id: str,
    service_account: str,
    schedule: str,
    time_zone: str = "America/Santiago",
    region: str = "us-central1",
) -> None:

    """
    Creates a scheduler job with a pipeline and its parameters as input
    """

    payload_bytes = json.dumps(payload).encode()

    client = scheduler.CloudSchedulerClient()

    parent = client.common_location_path(project_id, region)
    cloud_function_url = (
        f"https://{region}-{project_id}.cloudfunctions.net/pipeline-launcher"
    )
    headers = {
        "Content-Type": "application/octet-stream",
        "User-Agent": "Google-Cloud-Scheduler",
    }

    http_method = scheduler.HttpMethod.POST
    oidc_token = scheduler.OidcToken(
        service_account_email=service_account, audience=cloud_function_url
    )

    http_target = scheduler.HttpTarget(
        uri=cloud_function_url,
        http_method=http_method,
        headers=headers,
        body=payload_bytes,
        oidc_token=oidc_token,
    )

    job = scheduler.Job(
        name=os.path.join(parent, "jobs", name),
        description="",
        http_target=http_target,
        schedule=schedule,
        time_zone=time_zone,
    )

    request = scheduler.CreateJobRequest(parent=parent, job=job)

    client.create_job(request=request)
