import argparse
import glob
import json
import os
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score
from azureml.core import Model, Run
import joblib

def evaluate_model(model_name, model_version, test_data_path, outcome_label, output_path):
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # Load the training data from the CSV files
        print('Loacating test dataset files...')
        csv_files = glob.glob(os.path.join(test_data_path, '*.csv'))
        print(f'Found {csv_files.count} files in test dataset')

        print('Loading test dataset files...')
        df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        print(f'Loaded files in dataframe with schema:')
        print(df.info())

        # Split the test data into features and labels
        X_test = df.drop(outcome_label, axis=1)
        y_test = df[outcome_label]

        # Load the model from Azure ML
        ws = Run.get_context().experiment.workspace
        
        # Get the latest model if version is not provided
        # otherwise get the specific version
        if model_version is None or len(model_version) == 0 or model_version == 'latest':
            model = Model(ws, name=model_name)
        else:
            model = Model(ws, name=model_name, version=model_version)
        
        # download the model
        model_path = model.download(exist_ok=True)
        
        # Load the model from the downloaded file
        trained_model = joblib.load(model_path)

        # Predict on the test data
        y_pred = trained_model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1_score = 2 * (accuracy * recall) / (accuracy + recall)
        
        # Prepare the results
        metrics = {
            "model_name": model_name,
            "model_version": model_version,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
            "fpr": 1-precision,
            "fnr": 1-recall
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
