import argparse
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import mlflow


def download_model(ml_client, model_name, model_version, local_folder):
    """
    Download the model from Azure ML workspace.

    :param ml_client: The MLClient object
    :param model_name: The name for the Azure ML model
    :param model_version: The version for the Azure ML model
    :param local_folder: The local folder to save the model
    """

    # Download the model from Azure ML workspace
    ml_client.models.download(
        model_name=model_name,
        model_version=model_version,
        local_path=local_folder,
    )

    print(f"Model downloaded to {local_folder}")


def upload_model(ml_client, model_name, model_version, local_folder):
    """
    Upload the model to Azure ML workspace.

    :param ml_client: The MLClient object
    :param model_name: The name for the Azure ML model
    :param model_version: The version for the Azure ML model
    :param local_folder: The local folder to save the model
    """
    # Load the trained model
    print("Loading trained model...")
    model = mlflow.sklearn.load_model(local_folder)
    print(f"Loaded model: {model}")

    # Register the model
    print(f"Registering model as: {model_name}:{model_version}")
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=model_name,
        artifact_path=model_name,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    print(f"Model uploaded to Azure ML workspace")


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
        workspace_name=args.workspace_name,
    )

    # check workspace connection
    ws = ml_client.workspaces.get(args.workspace_name)
    print(
        f"Successfully connected to workspace: {ws.name} {ws.location} {ws.resource_group}"
    )

    return ml_client


def main(args):
    """
    Main function to perform the operation based on the arguments.

    :param args: The arguments containing the subscription_id, resource_group, workspace_name, model_name, model_version, local_folder, operation
    """

    # Get the MLClient object
    ml_client = get_ml_client(args)

    if args.operation == "download":
        # Download the model from Azure ML workspace
        ml_client.models.download(
            ml_client=ml_client,
            model_name=args.model_name,
            model_version=args.model_version,
            local_path=args.local_folder,
        )
        print(f"Model downloaded to {args.local_folder}")
    elif args.operation == "upload":
        # Upload the model to Azure ML workspace
        ml_client.models.upload(
            ml_client=ml_client,
            model_name=args.model_name,
            model_version=args.model_version,
            model_path=args.local_folder,
        )
        print(f"Model uploaded to Azure ML workspace")
    else:
        print(f"Invalid operation: {args.operation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subscription_id",
        type=str,
        help="The subscription id for the Azure ML workspace",
    )
    parser.add_argument(
        "--resource_group",
        type=str,
        help="The resource group for the Azure ML workspace",
    )
    parser.add_argument(
        "--workspace_name", type=str, help="The name for the Azure ML workspace"
    )
    parser.add_argument(
        "--model_name", type=str, help="The name for the Azure ML model"
    )
    parser.add_argument(
        "--model_version", type=str, help="The version for the Azure ML model"
    )
    parser.add_argument(
        "--local_folder", type=str, help="The local folder to save the model"
    )
    parser.add_argument(
        "--operation", choices=["download", "upload"], help="The operation to perform"
    )

    args = parser.parse_args()
    print("Printing received arguments...")
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")
    main(args)
