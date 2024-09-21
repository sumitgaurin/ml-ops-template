from azureml.core import Run, Model
import argparse
import mlflow

def register_model(model_name, model_path, model_version):
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # Get the workspace from the run
        ws = Run.get_context().experiment.workspace

        # Register the model
        model = Model.register(workspace=ws,
                            model_name=model_name,
                            model_path=model_path)
        
        # se the registered_model output variable
        model_version = model.version
        print(f"Model {model_name} registered successfully with version {model_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a model with Azure ML Workspace")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model folder")
    parser.add_argument("--model_version", type=str, required=True, help="Latest version of the registered model")

    args = parser.parse_args()
    
    register_model(args.model_name, args.model_path, args.model_version)