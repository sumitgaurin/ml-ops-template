import argparse
import json
import ast
from jsondiff import diff
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, load_environment, load_component

def compare_json(json1, json2) -> bool:
    """
    Compare two json objects for equality.

    :param json1: The first json object
    :param json2: The second json object
    :return: True if the json objects are equal, False otherwise
    """
    print(f"Comparing json objects...")
    print(f"Json1: {json1}")
    print(f"Json2: {json2}")    

    # Compare the json objects and return true if they are equal and there is no difference
    difference = diff(json1, json2)
    print(f"Diff: {difference}")
    return difference == {}


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
    Main function to call the Azure ML endpoint and validate result.

    :param args: The arguments containing the subscription_id, resource_group, workspace_name etc.
    :return: None
    """
    # Get the ml_client based on the credentials
    print(f"Connecting to Azure ML Service...")
    ml_client = get_ml_client(args)

    # Invoke the end point using the ml_client
    print(f"Invoking the endpoint '{args.endpoint_name}'...")
    scoring_response = ml_client.online_endpoints.invoke(
        endpoint_name=args.endpoint_name, 
        deployment_name=args.deployment_name,
        request_file=args.request_data
    )

    print("Response Received...")
    actual_result = ast.literal_eval(scoring_response)
    print("Response: ", actual_result)
    
    # Read the expected result from the response data file
    with open(args.response_data, "r") as file:
        expected_result = file.read()
    print("Expected Result: ", expected_result)

    # Compare the expected result with the actual result in json format
    assert compare_json(
        json.loads(actual_result), json.loads(expected_result)
    ), "The response data does not match the expected result"

    print("The response data matches the expected result")
    print("Validation successful!")


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
        "--endpoint_name", type=str, help="The name for the Azure ML endpoint"
    )
    parser.add_argument(
        "--deployment_name", type=str, help="The name for the Azure ML deployment"
    )
    parser.add_argument(
        "--request_data", type=str, help="The path to the request data json file"
    )
    parser.add_argument(
        "--response_data", type=str, help="The path to the response data json file"
    )

    args = parser.parse_args()
    print("Printing received arguments...")
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")
    main(args)
