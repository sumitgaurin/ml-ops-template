import argparse
import glob
import json
import os
import pandas as pd
import mlflow
from mlflow.sklearn import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model(model_id:str, model_path:str, test_data_path:str, outcome_label:str, result_file:str)->None:
    """
    Evaluate a machine learning model on test data and save the evaluation metrics.
    
    Parameters
    -----------
    model_id : str 
        Identifier for the model being evaluated.

    model_path : str 
        Path to the trained model file.

    test_data_path : str 
        Directory path containing test data CSV files.
    
    outcome_label : str
        The column name in the test data that contains the true labels.
    
    result_file : str
        Path to the file where evaluation metrics will be saved.
    
    Returns
    --------
    None : The function saves the evaluation metrics to the specified result file.
    """
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # Load the training data from the CSV files
        print('Loacating test dataset files...')
        csv_files = glob.glob(os.path.join(test_data_path, '*.csv'))
        print(f'Found {len(csv_files)} files in test dataset')

        print('Loading test dataset files...')
        df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        print(f'Loaded files in dataframe with schema:')
        print(df.info())

        # Split the test data into features and labels
        X_test = df.drop(outcome_label, axis=1)
        y_test = df[outcome_label]

        # Get the model and if not found then send the zero-metric
        metrics = {
            "model_id": "",
            "accuracy": 0,
            "recall": 0,
            "precision": 0,
            "f1_score": 0,
            "fpr": 1,
            "fnr": 1
        }

        # Load the model from the model path
        trained_model = load_model(model_path)

        # Run model evaluation only if the model for the specified version is found
        if trained_model is not None:
            # Predict on the test data
            y_pred = trained_model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1_score = 2 * (accuracy * recall) / (accuracy + recall)
            
            # Prepare the results
            metrics = {
                "model_id": model_id,
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
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'w') as report_file:
            report_file.write(json_output)

        print(f"Evaluation results:\n{json_output}")
        print(f"Results saved to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, help='A string identiifer which can be used to recognize the results')    
    parser.add_argument('--model_path', type=str, help='Path containing the trained model')    
    parser.add_argument('--test_data', type=str, help='Path to the test data CSV file')
    parser.add_argument('--outcome_label', type=str, help='Name of the column with the outcome label')
    parser.add_argument('--result_file', type=str, help='Path to save the results JSON file')

    args = parser.parse_args()
    print('Printing received arguments...')
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")
        
    evaluate_model(args.model_id, args.model_path, args.test_data, args.outcome_label, args.result_file)
