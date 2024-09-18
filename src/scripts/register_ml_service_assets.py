import argparse
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_environment, load_component
from azure.core.exceptions import ResourceNotFoundError

def asset_exists(asset_collection, asset_name, asset_version)->bool:
    """
    Check if the asset already exists in the workspace.

    :param asset_collection: The collection of assets to check
    :param asset_name: The name of the asset to check
    :param asset_version: The version of the asset to check
    :return: If it does, return True, else return False. If there is an error, print the error and return False.
    """ 
    try:
        _ = asset_collection.get(name=asset_name, version=asset_version)
        print(f"Asset '{asset_name}' with version '{asset_version}' exists.")
        return True
    except ResourceNotFoundError:
        print(f"Asset '{asset_name}' with version '{asset_version}' does not exist.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
    return False

def register_environments(ml_client, src_path, ignore_list=[]):
    """
    Register the custom environments in the workspace.

    :param ml_client: The MLClient object to use to register the environments
    :param src_path: The path to the source code folder
    :param ignore_list: The list of environment folder names to ignore
    :return: None
    """

    # Get the root environment definition folder
    env_root = os.path.join(src_path, "environments")

    # Find all the environment definition folders under it loop through them
    environments = [d for d in os.listdir(env_root) if os.path.isdir(os.path.join(env_root, d))]
    for env in environments:
        # If environment is in the ignore list, skip it
        if env in ignore_list:
            continue

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

def register_components(ml_client, src_path, ignore_list=[]):
    """
    Register the custom components in the workspace.

    :param ml_client: The MLClient object to use to register the components
    :param src_path: The path to the source code folder
    :param ignore_list: The list of component definition file names to ignore
    :return: None
    """

    # Get the root component definition folder
    comp_root = os.path.join(src_path, "components")

    # Find all the yaml recursively under the root folder
    # also ignore the file names in the ignore_list
    for root, dirs, files in os.walk(comp_root):
        for file in files:
            if file.endswith(".yaml") and file not in ignore_list:
                comp_path = os.path.join(root, file)
                # Load the component
                comp_asset = load_component(source=comp_path)
                # check if the asset already exists
                if not asset_exists(ml_client.components, comp_asset.name, comp_asset.version):
                    print(f"Registering component {comp_asset.name} ...")
                    # Create or update the component in the workspace
                    ml_client.components.create_or_update(comp_asset)
                    print(f"Component '{comp_asset.name}' registered in workspace.")

def get_ml_client(args) -> MLClient:
    """
    Get the MLClient object based on the DefaultAzureCredential authentication.
    For the authentication to work, session should be already logged in using AzureCLI.  

    :param args: The arguments containing the subscription_id, resource_group, workspace_name
    :return: The MLClient object
    """

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
    """
    Main function to register the Azure ML Service assets.

    :param args: The arguments containing the subscription_id, resource_group, workspace_name, src_path
    :return: None
    """
    # Get the ml_client based on the credentials
    print(f"Connecting to Azure ML Service...")
    ml_client = get_ml_client(args)

    print(f"Registering custom environments...")
    register_environments(ml_client, args.src_path)
    
    print(f"Registering custom components...")
    register_components(ml_client, args.src_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription_id', type=str, help='The subscription id for the Azure ML workspace')
    parser.add_argument('--resource_group', type=str, help='The resource group for the Azure ML workspace')
    parser.add_argument('--workspace_name', type=str, help='The name for the Azure ML workspace')
    parser.add_argument('--src_path', type=str, help='The path to the source code folder')

    args = parser.parse_args()
    main(args)
