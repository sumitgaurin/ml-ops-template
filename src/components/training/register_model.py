import argparse
import json
import mlflow
import mlflow.sklearn
import os


def log_report(report: list, log_entry: str) -> None:
    """
    Logs an entry to the report list.

    Parameters
    ----------
    report : list
        The report list to log the entry to.

    log_entry : str
        The entry to log to the report list.

    Returns
    -------
    None : The function logs the entry to the report list.
    """
    report.append(log_entry)
    print(log_entry)


def register_trained_model(
    comparison_report: str,
    model_path: str,
    model_name: str,
    model_id: str,
    register_report: str,
) -> None:
    """
    Registers a trained model with MLflow if the comparison report shows better results.

    Parameters
    ----------
    comparison_report : str
        The path to the comparison report generated during model training.

    model_path : str
        The path to the trained model.

    model_name : str
        The name of the model.

    model_id : str
        The key of the trained model in the comparison report.

    register_report : str
        The path to the report generated during model registration.

    Returns
    -------
    None : The function registers the trained model with MLflow.
    """
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # enable autologging
        mlflow.sklearn.autolog()

        print("Initializing model registration report...")
        registration_report = []

        log_report(registration_report, "Starting model registration...")
        # check if the comparison report exists
        if os.path.exists(comparison_report):
            # Load the json comparison report
            log_report(registration_report, "Loading json comparison report...")
            with open(comparison_report, "r") as f:
                cp_report = json.load(f)

            if cp_report["best_model_id"] == model_id:
                log_report(
                    registration_report, "Trained model is better than existing."
                )

                # Load the trained model
                log_report(registration_report, "Loading trained model...")
                model = mlflow.sklearn.load_model(model_path)
                log_report(registration_report, f"Loaded model: {model}")

                # Register the model
                log_report(registration_report, f"Registering model as: {model_name}")
                mlflow.sklearn.log_model(
                    sk_model=model,
                    registered_model_name=model_name,
                    artifact_path=model_name,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )
                log_report(registration_report, "Model registration complete.")
            else:
                log_report(
                    registration_report, "Existing model is better than trained model."
                )
                log_report(registration_report, "Model registration aborted.")
        else:
            log_report(registration_report, "Comparison report not found.")
            log_report(registration_report, "Model registration aborted.")

        # Save the report
        os.makedirs(os.path.dirname(register_report), exist_ok=True)
        print(f"Saving comparison report to: {register_report}")
        data = "\n".join(registration_report)
        with open(register_report, "w") as f:
            f.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison_report", type=str, help="Path to the json comparison report"
    )
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument(
        "--model_id", type=str, help="Key of the trained model in the comparison report"
    )
    parser.add_argument(
        "--register_report",
        type=str,
        help="Path to the report generated during model registration",
    )

    args = parser.parse_args()
    print("Printing received arguments...")
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")

    register_trained_model(
        args.comparison_report,
        args.model_path,
        args.model_name,
        args.model_id,
        args.register_report,
    )
