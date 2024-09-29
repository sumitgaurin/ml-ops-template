import argparse
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.core.exceptions import ResourceNotFoundError


def deploy_model(args):
    # Set up authentication
    credential = DefaultAzureCredential()

    # Initialize MLClient
    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    # Check if the endpoint exists
    endpoint_name = args.endpoint_name
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        print(f"Endpoint '{endpoint_name}' already exists.")
        if args.delete_if_existing:
            print(f"Deleting existing endpoint '{endpoint_name}'.")
            ml_client.online_endpoints.begin_delete(name=endpoint_name).wait()
            print(f"Endpoint '{endpoint_name}' deleted.")
            endpoint = None
        else:
            print(f"Using existing endpoint '{endpoint_name}'.")
    except ResourceNotFoundError:
        print(f"Endpoint '{endpoint_name}' does not exist. It will be created.")
        endpoint = None

    # Check if the workspace has a private endpoint enabled if public network access is disabled
    if not args.public_endpoint:
        workspace = ml_client.workspaces.get(args.workspace_name)
        if workspace.public_network_access:
            print(
                f"Workspace '{args.workspace_name}' does not have a private endpoint enabled. Exiting deployment."
            )
            return

    # Create endpoint if it doesn't exist
    if not endpoint:
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            auth_mode=args.auth_mode,
            public_network_access="enabled" if args.public_endpoint else "disabled",
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
        print(f"Endpoint '{endpoint_name}' created.")

    # Get the model
    if args.model_version:
        model = ml_client.models.get(name=args.model_name, version=args.model_version)
    else:
        # Get the latest version of the model
        models = ml_client.models.list(name=args.model_name)
        model = max(models, key=lambda m: m.version)
        print(f"No model version specified. Using the latest version: {model.version}")

    # Get the environment
    environment = ml_client.environments.get(name=args.environment_name)

    # Define the deployment
    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment=environment,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )

    # Add scoring script if provided
    if args.scoring_file:
        deployment.code_configuration = {
            "code": os.path.dirname(args.scoring_file),
            "scoring_script": os.path.basename(args.scoring_file),
        }

    # Create or update the deployment
    print(f"Creating or updating deployment '{args.deployment_name}'.")
    ml_client.online_deployments.begin_create_or_update(deployment).wait()
    print(f"Deployment '{args.deployment_name}' completed.")

    # Set the deployment as default
    ml_client.online_endpoints.begin_update(
        endpoint_name, default_deployment_name=args.deployment_name
    ).wait()
    print(
        f"Deployment '{args.deployment_name}' set as default for endpoint '{endpoint_name}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy a model as an endpoint in Azure ML."
    )

    # Required parameters
    parser.add_argument(
        "--subscription_id", type=str, required=True, help="Azure subscription ID"
    )
    parser.add_argument(
        "--resource_group", type=str, required=True, help="Azure resource group name"
    )
    parser.add_argument(
        "--workspace_name", type=str, required=True, help="Azure ML workspace name"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to deploy"
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        required=True,
        help="Name of the endpoint to create or update",
    )
    parser.add_argument(
        "--deployment_name", type=str, required=True, help="Name of the deployment"
    )
    parser.add_argument(
        "--auth_mode",
        type=str,
        choices=["key", "aml_token", "aad_token"],
        required=True,
        help="Authentication mode for the endpoint",
    )
    parser.add_argument(
        "--environment_name",
        type=str,
        required=True,
        help="Name of the Azure ML environment to use",
    )

    # Optional parameters with defaults
    parser.add_argument(
        "--model_version", type=str, default=None, help="Version of the model to deploy"
    )
    parser.add_argument(
        "--instance_type",
        type=str,
        default="Standard_DS3_v2",
        help="VM size for the deployment",
    )
    parser.add_argument(
        "--instance_count", type=int, default=1, help="Number of instances to deploy"
    )
    parser.add_argument(
        "--public_endpoint",
        action="store_true",
        help="Flag to enable public endpoint (default: false)",
    )
    parser.add_argument(
        "--delete_if_existing",
        action="store_true",
        help="Delete endpoint if it already exists",
    )

    # Optional parameters
    parser.add_argument(
        "--scoring_file",
        type=str,
        default=None,
        help="Path to the custom scoring script",
    )

    args = parser.parse_args()
    print("Printing received arguments...")
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")
    deploy_model(args)
