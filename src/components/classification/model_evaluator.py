import argparse
import json
import os
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, recall_score
from azureml.core import Model, Run
import joblib

def evaluate_model(model_name, model_version, test_data_path, outcome_label, output_path):
    # Start Logging
    mlflow.start_run()

    # Load the test data
    test_data = pd.read_csv(test_data_path)
    # Split the test data into features and labels
    X_test = test_data.drop(outcome_label, axis=1)
    y_test = test_data[outcome_label]

    # Load the model from Azure ML
    ws = Run.get_context().experiment.workspace
    model = Model(ws, name=model_name, version=model_version)
    model_path = model.download(exist_ok=True)
    
    # Load the model from the downloaded file
    trained_model = joblib.load(model_path)

    # Predict on the test data
    y_pred = trained_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = 2 * (accuracy * recall) / (accuracy + recall)
    
    # Prepare the results
    metrics = {
        "model_name": model_name,
        "model_version": model_version,
        "accuracy": accuracy,
        "recall": recall,
        "f1_score": f1_score
    }

    # Dump the final dictionary to a JSON string (or to a file)
    json_output = json.dumps(metrics, indent=4)

    # Save metrics to a JSON file for future comparison
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as report_file:
        report_file.write(json_output)

    print(f"Evaluation results:\n{json_output}")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--model_version', type=str, help='Version of the model')
    parser.add_argument('--test_data', type=str, help='Path to the test data CSV file')
    parser.add_argument('--output_path', type=str, help='Path to save the results JSON file')

    args = parser.parse_args()
    evaluate_model(args.model_name, args.model_version, args.test_data, args.output_path)
