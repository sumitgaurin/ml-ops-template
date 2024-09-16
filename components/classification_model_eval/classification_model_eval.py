import argparse
import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from azureml.core import Workspace, Model, Run
import joblib

def evaluate_model(model_name, model_version, test_data_path, output_dir):
    # Load the test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop('label', axis=1)  # Assuming 'label' is the target column
    y_test = test_data['label']

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

    # Prepare the results
    metrics = {
        "model_name": model_name,
        "model_version": model_version,
        "accuracy": accuracy,
    }

    # Save metrics to a JSON file for future comparison
    os.makedirs(output_dir, exist_ok=True)
    results_file_path = os.path.join(output_dir, 'metrics.json')
    with open(results_file_path, 'w') as f:
        json.dump(metrics, f)

    print(f"Results saved to {results_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--model_version', type=str, help='Version of the model')
    parser.add_argument('--test_data', type=str, help='Path to the test data CSV file')
    parser.add_argument('--output_dir', type=str, help='Directory to save the results')

    args = parser.parse_args()
    evaluate_model(args.model_name, args.model_version, args.test_data, args.output_dir)
