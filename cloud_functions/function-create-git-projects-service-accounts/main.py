import os
import json
import github
import tempfile
from github import Github, UnknownObjectException, GithubException
from cookiecutter.main import cookiecutter
from google.cloud import storage

github.enable_console_debug_logging()


def leer_archivo_bucket(bucket_name, source_blob_name):
        storage_client = storage.Client(project='ml-framework-config')
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
    
        return blob.download_as_text()
    
def create_github_project(file_event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """    
    # Configuración
    github_token = os.getenv('GITHUB_TOKEN_SECRET')
    plantilla_url = f'https://{github_token}@github.com/aquilesIIIMB/cookiecutter-data-science-template.git'
    nombre_usuario = 'aquilesIIIMB'
    email_usuario = 'aquiles.martinez@ug.uchile.cl'
    bucket_name = 'bucket-create-git-projects-service-accounts'
    config_input = json.loads(leer_archivo_bucket(bucket_name, file_event['name']))
    new_project_name = config_input['projectName']

    try:
        # Conexion a GitHub
        g = Github(github_token)
        usuario = g.get_user()

        # Verificar que existe el repositorio
        usuario.get_repo(new_project_name)

        return f'repository {new_project_name} already exists'
    except UnknownObjectException:
        try:
            # Conexion a GitHub
            g = Github(github_token)
            usuario = g.get_user()
        
            # Crear repositorio en GitHub
            usuario.create_repo(new_project_name, private=True)

        except GithubException as e:
            return f'repository creation error: {str(e)}'

    # Genera el proyecto con Cookiecutter
    with tempfile.TemporaryDirectory() as tmpdirname:
        cookiecutter(plantilla_url, no_input=True, overwrite_if_exists=True, output_dir=tmpdirname, extra_context=config_input)
        proyecto_dir = os.path.join(tmpdirname, new_project_name)

        # Inicializar Git, primer commit y push
        os.chdir(proyecto_dir)
        os.system('git init -b main')
        os.system(f'git config --global user.name "{nombre_usuario}"')
        os.system(f'git config --global user.email "{email_usuario}"')
        os.system('git add .')
        os.system('git commit -m "Primer commit"')
        os.system(f'git remote add origin https://{github_token}@github.com/aquilesIIIMB/{new_project_name}.git')
        os.system('git remote -v')
        os.system('git push -u origin main')

    return f'repository {new_project_name} is created'
