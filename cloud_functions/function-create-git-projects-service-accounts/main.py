import os
import json
import tempfile
import uuid
import re

from typing import Dict
from github import Github, UnknownObjectException, GithubException
from cookiecutter.main import cookiecutter

from google.cloud import storage
from google.cloud import bigquery
from google.oauth2 import service_account
import googleapiclient.discovery 
from google.api_core.exceptions import Conflict
import logging
from google.cloud import logging as gcp_logging


def setup_logging(
    log_level=logging.INFO
) -> None:
    """
    Set up Python logging with a custom logging level and send log messages to Google Cloud Logging.
    
    Args:
        log_level (int): The desired logging level, e.g., logging.DEBUG, logging.INFO, logging.WARNING, etc.
    """
    # Create a Google Cloud Logging client
    logging_client = gcp_logging.Client()
    # Get the default logging handler for Google Cloud Logging
    handler = logging_client.get_default_handler()
    # Get the root logger
    root_logger = logging.getLogger()
    # Set the logging level to INFO (you can adjust this level as needed)
    root_logger.setLevel(log_level)
    # Add the Google Cloud Logging handler to the root logger
    root_logger.addHandler(handler)


def sanitize_name(
    name: str
) -> str:
    """
    Sanitizes a given string by removing special characters and spaces, converting it to lowercase, 
    and ensuring it doesn't contain consecutive or trailing hyphens.

    Args:
        name (str): The string to be sanitized.

    Returns:
        str: A sanitized string suitable for use as a bucket or dataset name.

    Raises:
        ValueError: If the sanitized name is empty.
    """
    # Remove special characters and spaces, and convert to lowercase
    name = re.sub(r'[^a-zA-Z0-9-]', '', name.lower())
    # Remove consecutive hyphens and leading/trailing hyphens
    name = re.sub(r'-+', '-', name).strip('-')

    # Ensure the name is not empty
    if not name:
        raise ValueError("Invalid bucket name after sanitization")

    return name


def generate_bucket_name(
    app_name: str
) -> str:
    """
    Generates a unique bucket name by sanitizing the provided application name and appending a truncated UUID.

    Args:
        app_name (str): The name of the application for which the bucket is being created.

    Returns:
        str: A unique bucket name.
    """
    # Sanitize the app name
    sanitized_app_name = sanitize_name(app_name)
    # Generate a UUID4 code
    uuid_code = str(uuid.uuid4())
    # Maximum length for the UUID4 code to fit within 63 characters
    max_uuid_length = 63 - len(sanitized_app_name) - 1  # Subtract 1 for the hyphen

    if max_uuid_length < 1:
        # Create the bucket name using just the sanitized app name 
        bucket_name = sanitized_app_name[:63]
    else:
        # Truncate or limit the UUID4 code to fit within the maximum length
        truncated_uuid_code = uuid_code[:max_uuid_length]
        # Concatenate the sanitized app name and truncated UUID4 code
        bucket_name = f"{sanitized_app_name}-{truncated_uuid_code}"

    return bucket_name


def generate_dataset_name(
    app_name: str
) -> str:
    """
    Generates a unique dataset name by sanitizing the provided application name and appending a truncated UUID.

    Args:
        app_name (str): The name of the application for which the dataset is being created.

    Returns:
        str: A unique dataset name.
    """
    # Sanitize the app name
    sanitized_app_name = sanitize_name(app_name)

    # Create the dataset name using just the sanitized app name 
    dataset_name = sanitized_app_name[:1024]

    return dataset_name


def create_gcs_bucket(
    bucket_name: str, 
    project_id: str,
    app_name: str,
    git_project_name: str,
    bucket_location: str = "us-central1", 
    bucket_class: str = "STANDARD",
) -> None:
    """
    Creates a Cloud Storage Bucket in a Google Cloud project.

    Args:
    - bucket_name (str): The name of the Google Cloud Storage Bucket.
    - bucket_location (str): The GCP location to host the Google Cloud Storage Bucket. 
    - bucket_class (str): Kind of Google Cloud Storage Bucket depending on storage time. Possible values: STANDARD, NEARLINE, COLDLINE.
    """
    storage_client = storage.Client(project=project_id)
    try:
        bucket = storage.Bucket(storage_client, name=bucket_name)
        bucket.labels = {"application_name": app_name, "git_project": git_project_name}
        bucket.storage_class = bucket_class
        bucket = storage_client.create_bucket(bucket, project=project_id, location=bucket_location) 
        logging.info(f"Bucket {bucket.name} created.")
    except Exception as e:
        logging.info(f"Failed to create dataset: {e}")
    

def create_bq_dataset(
    dataset_name: str, 
    project_id: str,
    app_name: str,
    git_project_name: str,
    dataset_location="us-central1"
) -> None:
    """
    Creates a Bigquery dataset to save tables in a Google Cloud project.

    Args:
    - dataset_name (str): The name of the Bigquery Dataset.
    - dataset_location (str): The GCP location to host the Bigquery Dataset.
    """
    bigquery_client = bigquery.Client(project=project_id)
    try:
        dataset = bigquery.Dataset(bigquery_client.dataset(dataset_name))
        dataset.location = dataset_location
        dataset.labels = {"application_name": app_name, "git_project": git_project_name}
        dataset = bigquery_client.create_dataset(dataset, project=project_id)
        logging.info(f'Dataset {dataset.dataset_id} created.')
    except Exception as e:
        logging.info(f"Failed to create dataset: {e}")


def read_file_from_bucket(
    bucket_name: str, 
    source_blob_name: str
) -> str:
    """
    Reads and returns the content of a blob from a specified Google Cloud Storage bucket.

    Args:
    - bucket_name (str): The name of the Google Cloud Storage bucket.
    - source_blob_name (str): The name of blob (file) in the bucket to be read.

    Returns:
    - str: The text content of blob.
    """
    storage_client = storage.Client(project='ml-framework-config')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    return blob.download_as_text()


def create_service_account_ml_framework_projects(
    account_name: str, account_description: str,
    project_id: str, service_account_key_json: Dict
) -> None:
    """
    Creates a new service account in a Google Cloud project.

    Args:
    - account_name (str): The name to assign to the new service account.
    - account_description (str): A brief description of the service account's purpose.
    - project_id (str): The Google Cloud project ID where the service account will 
        be created.
    - service_account_key_json (Dict): The service account key (credentials) for 
        authenticating the request, provided as a dictionary.
    """
    credentials = service_account.Credentials.from_service_account_info(
        service_account_key_json
    )
    client = googleapiclient.discovery.build("iam", "v1", credentials=credentials)
    created_account = (
        client.projects()
        .serviceAccounts()
        .create(
            name="projects/" + project_id,
            body={
                "accountId": account_name, 
                "serviceAccount": {
                    "displayName": account_name,
                    "description": account_description
                }
            }
        )
        .execute()
    )

    logging.info(f'Service account created: {created_account["email"]}')


def create_github_project_using_cookiecutter(
    github_token: str, 
    new_project_name: str, 
    template_url: str,
    config_input: Dict, 
    user_name: str, 
    user_email: str
) -> int:
    """
    Creates a new GitHub repository for a project, using the cookiecutter template.

    Args:
    - github_token (str): The GitHub token for authentication.
    - new_project_name (str): The name for the new GitHub repository.
    - config_input (Dict): Configuration data to pass to cookiecutter for 
        template customization.
    - user_name (str): GitHub user name for setting up the repository.
    - user_email (str): GitHub user email for setting up the repository.
    """
    try:
        g = Github(github_token)
        user = g.get_user()
        user.get_repo(new_project_name)
        logging.info(f'Repository {new_project_name} already exists')
        return 0
    except UnknownObjectException:
        try:
            g = Github(github_token)
            user = g.get_user()
            user.create_repo(new_project_name, private=True)
        except GithubException as e:
            logging.info(f'Repository creation error: {str(e)}')
            return 0

    with tempfile.TemporaryDirectory() as tmpdirname:
        cookiecutter(
            template_url, no_input=True, overwrite_if_exists=True,
            output_dir=tmpdirname, extra_context=config_input
        )
        project_dir = os.path.join(tmpdirname, new_project_name)
        os.chdir(project_dir)
        os.system('git init -b main')
        os.system(f'git config --global user.name "{user_name}"')
        os.system(f'git config --global user.email "{user_email}"')
        os.system('git add .')
        os.system('git commit -m "Initial commit"')
        os.system(
            f'git remote add origin https://{github_token}@github.com/'
            f'aquilesIIIMB/{new_project_name}.git'
        )
        os.system('git remote -v')
        os.system('git push -u origin main')
        os.system('git branch stage')
        os.system('git checkout stage')
        os.system('git branch')
        os.system('git push -u origin stage')

        logging.info(f'Repository {new_project_name} was created')
        return 1


def create_github_project_with_service_accounts(
    file_event: Dict, 
    context: Dict
) -> None:
    """
    Orchestrates the creation of a new GitHub project with associated Google Cloud service accounts and resources. 
    It reads configuration from a Cloud Storage bucket, uses cookiecutter for project initialization, 
    creates a new GitHub repository, and sets up Google Cloud service accounts and resources like GCS buckets and BigQuery datasets.

    Args:
        file_event (Dict): The event payload containing information about the change in the Cloud Storage bucket.
        context (Dict): Metadata for the event.

    Note:
        This function is designed to be triggered by a change to a specified Cloud Storage bucket.
    """
    github_token = os.getenv('GITHUB_TOKEN_SECRET')
    template_url = (
        f'https://{github_token}@github.com/'
        f'aquilesIIIMB/cookiecutter-data-science-template.git'
    )
    user_name = 'aquilesIIIMB'
    user_email = 'aquiles.martinez@ug.uchile.cl'
    bucket_name = 'bucket-create-git-projects-b89ce4a5-a3a8-486d-a156-4366af1fe5dc'
    config_input = json.loads(
        read_file_from_bucket(bucket_name, file_event['name'])
    )
    new_project_name = config_input['projectName']
    new_application_name = config_input['applicationName']
    credential_maas_json = json.loads(
        os.getenv('GOOGLE_APPLICATION_CREDENTIALS_MAAS')
    )

    maas_project_id = 'ml-framework-maas'
    new_service_account_maas_name = f'app-{str(uuid.uuid4())[:26]}'
    bucket_name = generate_bucket_name(new_application_name)
    dataset_name = generate_dataset_name(new_application_name)

    description_new_service_account_maas = (
        f'Service account to manage the implementation of '
        f'{new_application_name} located in '
        f'https://github.com/aquilesIIIMB/{new_project_name}.git'
    )

    config_input['serviceAccountMaasName'] = (
        f"{new_service_account_maas_name}@{maas_project_id}.iam.gserviceaccount.com"
    )
    config_input['serviceAccountExplorationName'] = ""
    config_input['serviceAccountDiscoveryName'] = ""

    config_input['bucketMaasName'] = bucket_name
    config_input['bucketExplorationName'] = ""
    config_input['bucketDiscoveryName'] = ""

    config_input['datasetMaasName'] = dataset_name
    config_input['datasetExplorationName'] = ""
    config_input['datasetDiscoveryName'] = ""

    setup_logging()

    if create_github_project_using_cookiecutter(
        github_token, new_project_name, template_url, 
        config_input, user_name, user_email
    ):

        create_service_account_ml_framework_projects(
            new_service_account_maas_name, 
            description_new_service_account_maas, 
            maas_project_id, credential_maas_json
        )

        create_gcs_bucket(
            bucket_name,
            maas_project_id,
            new_application_name,
            new_project_name,
        )

        create_bq_dataset(
            dataset_name,
            maas_project_id.
            new_application_name,
            new_project_name,
        )
