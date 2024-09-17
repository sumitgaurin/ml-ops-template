import argparse
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_environment
from azure.ai.ml.entities import Environment
from azure.core.exceptions import ResourceNotFoundError

def asset_exists(asset_collection, asset_name, asset_version)->bool:
    try:
        asset_collection.get(name=asset_name, version=asset_version)
        print(f"Asset '{asset_name}' with version '{asset_version}' exists.")
        return True
    except ResourceNotFoundError:
        print(f"Asset '{asset_name}' with version '{asset_version}' does not exist.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
    return False

def register_environments(ml_client, src_path):
    # Get the root environment definition folder
    env_root = os.path.join(src_path, "environments")

    # Find all the environment definition folders under it loop through them
    environments = [d for d in os.listdir(env_root) if os.path.isdir(os.path.join(env_root, d))]
    for env in environments:
        # ASSUMPTION: Each environment folder has a file called "definition.yaml" which
        # contains the environment defintion
        definition_path = os.path.join(env_root, env, "definition.yaml")
        if not os.path.isfile(definition_path):
            raise FileNotFoundError

        # Load the definition
        env_asset = load_environment(source=definition_path)
        # check if the asset already exists
        if not asset_exists(ml_client.environments, env_asset.name, env_asset.version):
            print(f"Registering environment {env_asset.name} ...")
            # Create or update the environment in the workspace
            ml_client.environments.create_or_update(env_asset)
            print(f"Environment '{env_asset.name}' registered in workspace.")

def get_ml_client(args) -> MLClient:
    # Authenticate using DefaultAzureCredential
    # Should be already logged in using AzureCLI in the session
    credential = DefaultAzureCredential()

    # Create MLClient
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name
    )
    
    # check workspace connection
    ws = ml_client.workspaces.get(args.workspace_name)
    print(f"Successfully connected to workspace: {ws.name} {ws.location} {ws.resource_group}")

    return ml_client

def main(args):
    # Get the ml_client based on the credentials
    print(f"Connecting to Azure ML Service...")
    ml_client = get_ml_client(args)
    print(f"Registering custom environments...")
    register_environments(ml_client, args.src_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription_id', type=str, help='subscription_id')
    parser.add_argument('--resource_group', type=str, help='resource_group')
    parser.add_argument('--workspace_name', type=str, help='workspace_name')
    parser.add_argument('--src_path', type=str, help='Full path for src folder')

    args = parser.parse_args()
    main(args)
