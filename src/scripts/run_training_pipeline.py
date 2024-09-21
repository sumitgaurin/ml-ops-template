import argparse
from ast import parse
from datetime import datetime
import json
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_job
#from src.components.helper import print_args

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
    Main function to run the paynet training pipeline.

    :param args: The arguments containing the subscription_id, resource_group, workspace_name, src_path
    :return: None
    """
    # Get the ml_client based on the credentials
    print(f"Connecting to Azure ML Service...")
    ml_client = get_ml_client(args)

    # Load the json data in dictionary format
    with open(args.pipeline_parameter_path, "r") as json_file:
        pipeline_parameters = json.load(json_file)
    print("loaded pipeline parameters...")
    print(pipeline_parameters)

    # Convert the dictionary into input override list
    transform_pipeline_parameters = [
        {f"inputs.{key}": value} for key, value in pipeline_parameters.items()
    ]
    print("transformed pipeline parameters...")
    print(transform_pipeline_parameters)

    # Load the pipeline job from the YAML file and override the input parameters    
    pipeline_job = load_job(
        source=args.pipeline_definition_path,
        params_override=transform_pipeline_parameters
    )

    # Modify the name of the job
    # If you are trying to create a new job, use a different name. If you are trying to update an existing job, 
    # the existing job's Jobs cannot be changed. 
    # Only description, tags, displayName, properties, and isArchived can be updated.
    dt_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    pipeline_job.name = f"{pipeline_job.name}_{dt_stamp}"
    print("loaded pipeline definition with parameter override...")
    print(pipeline_job)

    # Now the pipeline is ready for execution
    # Submit the job
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, 
        experiment_name=args.experiment_name
    )
    
    print(f"Pipeline job submitted. Job ID: {pipeline_job.name}")
    print(f"Pipeline job can be tracked by {pipeline_job.studio_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subscription_id', type=str, help='The subscription id for the Azure ML workspace')
    parser.add_argument('--resource_group', type=str, help='The resource group for the Azure ML workspace')
    parser.add_argument('--workspace_name', type=str, help='The name for the Azure ML workspace')
    parser.add_argument('--pipeline_definition_path', type=str, help='The path of the pipeline definition file')
    parser.add_argument('--pipeline_parameter_path', type=str, help='The path of the pipeline parameter file')
    parser.add_argument('--experiment_name', type=str, help='The name of the experiment under which the pipeline will run')
    

    args = parser.parse_args()
    #print_args(args)
    main(args)