import argparse
import os
from azureml.core import Run, Model
import joblib
import glob
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier

def train_model_paynet(model_name, training_data_path, model_version, output_model_path):
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # enable autologging
        mlflow.sklearn.autolog()

        # Load the training data from the CSV files
        print('Loacating training dataset files...')
        csv_files = glob.glob(os.path.join(training_data_path, '*.csv'))
        print(f'Found {csv_files.count} files in training dataset')

        print('Loading training dataset files...')
        df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        print(f'Loaded files in dataframe with schema:')
        print(df.info())
        
        ########################################################################
        # Placeholder for developer to write feature engineering based on the actual implementation
        # Example: 
        # Assume the target column is named 'label' and the rest are features
        X_train = df.drop(columns=['label'])
        y_train = df['label']
        
        # Initialize a Gradient Boosting Classifier (or any other classifier)
        model = GradientBoostingClassifier()
        
        # Train the model
        model.fit(X_train, y_train)    
        ########################################################################
        ########################################################################

        # Save the model to the output directory
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        joblib.dump(model, output_model_path)

        # Get the workspace from the run
        ws = Run.get_context().experiment.workspace
        # Register the model with the workspace
        latest_model = Model.register(workspace=ws,
                            model_name=model_name,
                            model_path=output_model_path)
        model_version = latest_model.version

        print(f"Model {model_name}:{model_version} saved at {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Define arguments for the script
    parser.add_argument('--model_name', type=str, help='Name of the trained model')
    parser.add_argument('--training_data', type=str, help='Path to the training data CSV file')
    parser.add_argument('--model_version', type=str, help='Registerd model version with the workspace')
    parser.add_argument('--output_model', type=str, help='Path of the trained model file')
    
    args = parser.parse_args()

    # Train the model and save it to the output directory
    train_model_paynet(args.model_name, args.training_data, args.model_version, args.output_model)
