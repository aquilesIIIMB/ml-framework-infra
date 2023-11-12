import os
import json
import tempfile
import uuid

from typing import Dict
from github import Github, UnknownObjectException, GithubException
from cookiecutter.main import cookiecutter

from google.cloud import storage, iam
from google.oauth2 import service_account


def read_file_from_bucket(bucket_name: str, source_blob_name: str) -> str:
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
) -> str:
    """
    Creates a new service account in a Google Cloud project and returns its name.

    Args:
    - account_name (str): The name to assign to the new service account.
    - account_description (str): A brief description of the service account's purpose.
    - project_id (str): The Google Cloud project ID where the service account will 
        be created.
    - service_account_key_json (Dict): The service account key (credentials) for 
        authenticating the request, provided as a dictionary.

    Returns:
    - str: Confirmation message including the name of the created service account.
    """
    credentials = service_account.Credentials.from_service_account_info(
        service_account_key_json
    )
    service = iam.ServiceAccountServiceClient(credentials=credentials)
    parent = f"projects/{project_id}"
    account_id = f"{account_name}@{project_id}.iam.gserviceaccount.com"
    service_account_info = {
        "account_id": account_name,
        "service_account": {
            "display_name": account_name,
            "description": account_description
        }
    }
    request = iam.CreateServiceAccountRequest(
        parent=parent,
        service_account=service_account_info,
        service_account_id=account_id
    )
    created_account = service.create_service_account(request=request)

    return f"Service account created: {created_account.name}"


def create_github_project_using_cookiecutter(
    github_token: str, new_project_name: str, 
    config_input: Dict, user_name: str, user_email: str
) -> str:
    """
    Creates a new GitHub repository for a project, using the cookiecutter template.

    Args:
    - github_token (str): The GitHub token for authentication.
    - new_project_name (str): The name for the new GitHub repository.
    - config_input (Dict): Configuration data to pass to cookiecutter for template customization.
    - user_name (str): GitHub user name for setting up the repository.
    - user_email (str): GitHub user email for setting up the repository.

    Returns:
    - str: A message indicating the result of the operation (repository creation or error message).
    """
    try:
        g = Github(github_token)
        user = g.get_user()
        user.get_repo(new_project_name)
        
        return f'Repository {new_project_name} already exists'
    except UnknownObjectException:
        try:
            g = Github(github_token)
            user = g.get_user()
            user.create_repo(new_project_name, private=True)
        except GithubException as e:
            return f'Repository creation error: {str(e)}'

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
            f'{new_project_name}.git'
        )
        os.system('git remote -v')
        os.system('git push -u origin main')
        
        return f'Repository {new_project_name} was created'


def create_github_project_with_service_accounts(
    file_event: Dict, context: Dict
) -> None:
    """
    Responds to a Cloud Storage bucket event to create a GitHub project with service accounts.

    This function is triggered by a change to a specified Cloud Storage bucket. It reads
    configuration data from the bucket, uses cookiecutter to initialize a project, and
    pushes it to a new GitHub repository. Additionally, it creates a new Google Cloud
    service account for the project.

    Args:
    - file_event (Dict): The event payload containing information about the change in the bucket.
    - context (Dict): Metadata for the event.
    """
    github_token = os.getenv('GITHUB_TOKEN_SECRET')
    template_url = (
        f'https://{github_token}@github.com/'
        f'aquilesIIIMB/cookiecutter-data-science-template.git'
    )
    user_name = 'aquilesIIIMB'
    user_email = 'aquiles.martinez@ug.uchile.cl'
    bucket_name = 'bucket-create-git-projects-service-accounts'
    config_input = json.loads(
        read_file_from_bucket(bucket_name, file_event['name'])
    )
    new_project_name = config_input['projectName']
    new_application_name = config_input['applicationName']
    credential_maas_json = json.loads(
        os.getenv('GOOGLE_APPLICATION_CREDENTIALS_MAAS')
    )
    maas_project_id = 'ml-framework-maas'
    new_service_account_maas_name = 'app-' + str(uuid.uuid4())[:26]
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


    create_github_project_using_cookiecutter(
        github_token, new_project_name, config_input, 
        user_name, user_email
    )

    create_service_account_ml_framework_projects(
        new_service_account_maas_name, 
        description_new_service_account_maas, 
        maas_project_id, credential_maas_json
    )
