from azureml.core import Run, Model
import argparse
import mlflow

def register_model(model_name, model_path, registered_model):
    # Start Logging
    mlflow.start_run()

    # Get the workspace from the run
    ws = Run.get_context().experiment.workspace

    # Register the model
    model = Model.register(workspace=ws,
                           model_name=model_name,
                           model_path=model_path)
    
    # se the registered_model output variable
    registered_model = f"{model.name}:{model.version}"
    print(f"Model {registered_model} registered successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a model with Azure ML Workspace")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model folder")
    parser.add_argument("--registered_model", type=str, required=True, help="Name of the registered model in name:version format")

    args = parser.parse_args()
    
    register_model(args.model_name, args.model_path, args.registered_model)