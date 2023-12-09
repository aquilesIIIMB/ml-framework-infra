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
from google.cloud.devtools.cloudbuild_v2.types import Connection, Repository, GitHubConfig, OAuthCredential, CreateConnectionRequest, CreateRepositoryRequest
from google.cloud.devtools.cloudbuild_v2.services.repository_manager import RepositoryManagerClient
import google.auth
from google.auth.transport.requests import Request
import requests



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


def create_github_connection(
    project_id: str,
    github_connection_name: str,
    github_token_secret_manager: str,
    app_installation_id: str,
    service_account_key_json: Dict
) -> CreateConnectionRequest:
    """
    Creates a new connection between Google Cloud Build and a GitHub repository.

    This function utilizes Google Cloud's Repository Manager API to establish a connection
    with GitHub, authenticated via OAuth credentials stored in Google Cloud Secret Manager.

    Parameters:
    - project_id (str): The unique identifier for your Google Cloud project.
    - github_connection_name (str): The name for the new GitHub connection in Cloud Build.
    - github_token_secret_manager (str): The resource identifier of the OAuth token in Google Cloud Secret Manager.
    - app_installation_id (str): The unique identifier for the GitHub app installation.

    Returns:
    - CreateConnectionResponse: The response object from the Cloud Build API with the details
      of the created connection.
    ```
    """
    github_config_credentials = OAuthCredential(
        oauth_token_secret_version=github_token_secret_manager
    )
    github_config = GitHubConfig(
        app_installation_id=app_installation_id, 
        authorizer_credential=github_config_credentials
    )
    
    cloud_build_connection = Connection(
        name=f'projects/{project_id}/locations/us-central1/connections/{github_connection_name}', 
        github_config=github_config
    )
    
    github_create_connection_request = CreateConnectionRequest(
        parent=f'projects/{project_id}/locations/us-central1',
        connection=cloud_build_connection,
        connection_id=github_connection_name
    )
    
    client_cb = RepositoryManagerClient.from_service_account_info(
        service_account_key_json
    )
    create_connection_request = client_cb.create_connection(request=github_create_connection_request)
    
    logging.info(create_connection_request.result())
    
    return create_connection_request


def create_cloud_build_repository(
    project_id: str,
    new_project_name: str,
    github_connection_name: str,
    cloud_build_repository_name: str,
    service_account_key_json: Dict
) -> CreateRepositoryRequest:
    """
    Creates a new repository in Google Cloud Build that is linked to a GitHub repository.

    This function sets up a repository in Cloud Build, allowing Cloud Build to interact with
    a specified GitHub repository. It is useful for setting up CI/CD pipelines for projects hosted on GitHub.

    Parameters:
    - project_id (str): The unique identifier for your Google Cloud project.
    - new_project_name (str): The name of the new project/repository in GitHub.
    - github_connection_name (str): The name of the existing GitHub connection in Cloud Build.
    - cloud_build_repository_name (str): The name you want to assign to the new repository in Cloud Build.

    Returns:
    - CreateRepositoryResponse: The response object from the Cloud Build API with the details
      of the created repository.
    ```
    """
    cloud_build_repository = Repository(
        name=f'projects/{project_id}/locations/us-central1/repositories/{cloud_build_repository_name}', 
        remote_uri=f"https://github.com/aquilesIIIMB/{new_project_name}.git"
    )
    
    github_create_repository_request = CreateRepositoryRequest(
        parent=f'projects/{project_id}/locations/us-central1/connections/{github_connection_name}',
        repository=cloud_build_repository,
        repository_id=cloud_build_repository_name
    )
    
    client_cb = RepositoryManagerClient.from_service_account_info(
        service_account_key_json
    )
    create_repository_request = client_cb.create_repository(request=github_create_repository_request)
    
    logging.info(create_repository_request.result())
    
    return create_repository_request


def create_github_trigger(
    project_id: str, 
    trigger_name: str,
    github_connection_name: str,
    cloud_build_repository_name: str,
    branch_pattern: str,
    branch_pattern_filter: str,
    build_config_path: str,
    service_account_email: str,
    description: str,
    service_account_key_json: Dict
) -> int:
    """
    Creates a trigger in Google Cloud Build for a GitHub repository.

    This function sets up a trigger in Cloud Build that responds to events (like push or pull request)
    on a specified branch of a GitHub repository linked via a Cloud Build connection.

    Parameters:
    - project_id (str): The unique identifier of the Google Cloud project.
    - trigger_name (str): The name for the new trigger in Cloud Build.
    - github_connection_name (str): The name of the GitHub connection in Cloud Build.
    - cloud_build_repository_name (str): The name of the repository in Cloud Build.
    - branch_pattern (str): The branch name or pattern to which the trigger will respond.
    - branch_pattern_filter (str): The type of event to respond to ('pull_request' or 'push').
    - build_config_path (str): Path to the build configuration file in the repository.
    - service_account (str): Service account to be used for the build.
    - description (str): A brief description of the trigger.

    Returns:
    - str: A success message if the trigger is created successfully, or an error message otherwise.
    ```
    """
    # Load credentials from service account JSON
    credentials = service_account.Credentials.from_service_account_info(
        service_account_key_json,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())

    url = f'https://cloudbuild.googleapis.com/v1/projects/{project_id}/locations/us-central1/triggers'

    if branch_pattern_filter == "pull_request":
        payload = {
            "name": trigger_name,
            "description": description,
            "filename": build_config_path,
            "serviceAccount": service_account_email,
            "includeBuildLogs": "INCLUDE_BUILD_LOGS_WITH_STATUS",
            "repositoryEventConfig": {
                "repository": f'projects/{project_id}/locations/us-central1/connections/{github_connection_name}/repositories/{cloud_build_repository_name}',
                "repositoryType": "GITHUB",
                "pullRequest": {
                    "commentControl": "COMMENTS_DISABLED",
                    "branch": branch_pattern
                }
            }
        }
    elif branch_pattern_filter == 'push':
        payload = {
            "name": trigger_name,
            "description": description,
            "filename": build_config_path,
            "serviceAccount": service_account_email,
            "includeBuildLogs": "INCLUDE_BUILD_LOGS_WITH_STATUS",
            "repositoryEventConfig": {
                "repository": f'projects/{project_id}/locations/us-central1/connections/{github_connection_name}/repositories/{cloud_build_repository_name}',
                "repositoryType": "GITHUB",
                "push": {
                    "branch": branch_pattern
                }
            }
        }
    else:
        raise ValueError("Invalid branch_filter value. It could be pull_request or push")

    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        logging.info("GitHub trigger created successfully")
        return 1
    else:
        logging.info(f"Error creating GitHub trigger. Status Code: {response.status_code}")
        logging.info(f"Error creating GitHub trigger. Message: {response.text}")
        return 0


def create_branch_protection(
    branch: str, 
    user_name: str, 
    new_project_name: str,
    github_token: str
) -> None:
    # Authentication
    g = Github(github_token)
    # Get the repository
    repo = g.get_repo(f"{user_name}/{new_project_name}")

    # Setting up branch protection rules
    protection_settings = {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews": True,
        "require_code_owner_reviews": True,
        "required_conversation_resolution": True,
        "lock_branch": True
    }
    
    # Apply protection
    repo.get_branch(branch).edit_protection(**protection_settings)
    
    logging.info(f"Protection applied to branch '{branch}' in '{repo.full_name}'")


def add_admin_role(
    collaborator_account: str, 
    user_name: str, 
    new_project_name: str,
    github_token: str
) -> None:
    # Authentication
    g = Github(github_token)
    # Get the repository
    repo = g.get_repo(f"{user_name}/{new_project_name}")

    try:
        # Add collaborator with admin access
        repo.add_to_collaborators(collaborator_account, permission="admin")
        print(f"User {collaborator_account} has been added as an admin to the repository.")
    except Exception as e:
        print(f"An error occurred: {e}")


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
    name = re.sub(r'[^a-zA-Z0-9-_]', '', name.lower())

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
    sanitized_app_name = sanitized_app_name.replace('_', '-')
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
    sanitized_app_name = sanitized_app_name.replace('-', '_')

    # Create the dataset name using just the sanitized app name 
    dataset_name = sanitized_app_name[:1024]

    return dataset_name


def create_gcs_bucket(
    bucket_name: str, 
    project_id: str,
    app_name: str,
    git_project_name: str,
    service_account_key_json: Dict,
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
    storage_client = storage.Client.from_service_account_info(
        service_account_key_json
    )
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
    service_account_key_json: Dict,
    dataset_location="us-central1"
) -> None:
    """
    Creates a Bigquery dataset to save tables in a Google Cloud project.

    Args:
    - dataset_name (str): The name of the Bigquery Dataset.
    - dataset_location (str): The GCP location to host the Bigquery Dataset.
    """
    bigquery_client = bigquery.Client.from_service_account_info(
        service_account_key_json
    )
    try:
        dataset = bigquery.Dataset(bigquery_client.dataset(dataset_name, project=project_id))
        dataset.location = dataset_location
        dataset.labels = {"application_name": app_name, "git_project": git_project_name}
        dataset = bigquery_client.create_dataset(dataset)
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
    new_application_name: str,
    service_account: str,
    github_token_secret_manager: str,
    app_installation_id: str,
    project_id: str,
    template_url: str,
    config_input: Dict, 
    user_name: str, 
    user_email: str,
    service_account_key_json: Dict
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
            user.create_repo(new_project_name, private=False)
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

        # Create stage branch in the new Github repo
        os.system('git branch stage')
        os.system('git checkout stage')
        os.system('git branch')
        os.system('git push -u origin stage')

        # Create mvp branch in the new Github repo
        os.system('git branch mvp')
        os.system('git checkout mvp')
        os.system('git branch')
        os.system('git push -u origin mvp')

        create_branch_protection(
            "main", 
            user_name,
            new_project_name,
            github_token
        )

        create_branch_protection(
            "stage", 
            user_name,
            new_project_name,
            github_token
        )

        # Create a cloud build connection to github and define triggers for each branch
        # Trigger parameters
        github_connection_name = f"{new_project_name}" 
        cloud_build_repository_name = f"{new_project_name}"

        mvp_trigger_name = f"trigger-mvp-{new_project_name}"
        mvp_branch_pattern_filter = 'pull_request'
        mvp_branch_pattern = "^stage$"
        mvp_description = "Trigger to check artifacts created from mvp folder"
        mvp_build_config_path = "maas/mvp/cloudbuild.yaml"

        stage_trigger_name = f"trigger-stage-{new_project_name}"
        stage_branch_pattern_filter = 'push'
        stage_branch_pattern = "^stage$"
        stage_description = "Trigger to generate artifacts in stage"
        stage_build_config_path = f"maas/{new_application_name}/cloudbuild.yaml"

        main_trigger_name = f"trigger-main-{new_project_name}"
        main_branch_pattern_filter = 'push'
        main_branch_pattern = "^main$"
        main_description = "Trigger to generate artifacts in main"
        main_build_config_path = f"maas/{new_application_name}/cloudbuild.yaml"

        # Create a github connection to read the repositories in github from cloud build
        create_github_connection(
            project_id,
            github_connection_name,
            github_token_secret_manager,
            app_installation_id,
            service_account_key_json
        )
        # Create a cloud build repository from a github repostiroy in the github connection 
        create_cloud_build_repository(
            project_id,
            new_project_name,
            github_connection_name,
            cloud_build_repository_name,
            service_account_key_json
        )
        # Create github trigger for mvp branch
        create_github_trigger(
            project_id,
            mvp_trigger_name,
            github_connection_name,
            cloud_build_repository_name,
            mvp_branch_pattern,
            mvp_branch_pattern_filter,
            mvp_build_config_path,
            service_account,
            mvp_description,
            service_account_key_json
        )
        # Create github trigger for stage branch
        create_github_trigger(
            project_id,
            stage_trigger_name,
            github_connection_name,
            cloud_build_repository_name,
            stage_branch_pattern,
            stage_branch_pattern_filter,
            stage_build_config_path,
            service_account,
            stage_description,
            service_account_key_json
        )
        # Create github trigger for main branch
        create_github_trigger(
            project_id,
            main_trigger_name,
            github_connection_name,
            cloud_build_repository_name,
            main_branch_pattern,
            main_branch_pattern_filter,
            main_build_config_path,
            service_account,
            main_description,
            service_account_key_json
        )

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
    adminAccounts_list = config_input['adminAccounts'].split(',')
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

    maas_github_token_secret_manager='projects/1099093996594/secrets/github-token-to-connect-cloudbuild/versions/latest'
    maas_app_installation_id='43923583'

    setup_logging()

    if create_github_project_using_cookiecutter(
        github_token, new_project_name, new_application_name, config_input['serviceAccountMaasName'],
        maas_github_token_secret_manager, maas_app_installation_id, maas_project_id, 
        template_url, config_input, 
        user_name, user_email, credential_maas_json
    ):
        for collaborator_account in adminAccounts_list:
            add_admin_role(
                collaborator_account,
                user_name,
                new_project_name,
                github_token,
            )

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
            credential_maas_json
        )

        create_bq_dataset(
            dataset_name,
            maas_project_id,
            new_application_name,
            new_project_name,
            credential_maas_json
        )
