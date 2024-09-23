import argparse
import os
import glob
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier

def train_model_paynet(training_data_path, outcome_label, output_model_path, n_estimators, learning_rate):
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # enable autologging
        mlflow.sklearn.autolog()

        # Load the training data from the CSV files
        print('Loacating training dataset files...')
        csv_files = glob.glob(os.path.join(training_data_path, '*.csv'))
        print(f'Found {len(csv_files)} files in training dataset')

        print('Loading training dataset files...')
        df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        print(f'Loaded files in dataframe with schema:')
        print(df.info())
        
        ########################################################################
        # Placeholder for developer to write feature engineering based on the actual implementation
        # Example: 
        # Assume the target column is named 'label' and the rest are features
        X_train = df.drop(columns=[outcome_label])
        y_train = df[outcome_label]
        
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate
        ).fit(X_train, y_train)
        
        # Train the model
        model.fit(X_train, y_train)    
        ########################################################################
        ########################################################################

        # Save the model to the output directory
        os.makedirs(output_model_path, exist_ok=True)
        # Saving the model to a file
        mlflow.sklearn.save_model(
            sk_model=model,
            path=output_model_path,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        print(f"SKLEARN model saved at {output_model_path} in CLOUDPICKLE format.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    # Define arguments for the script
    parser.add_argument('--training_data', type=str, help='Path to the training data CSV file')
    parser.add_argument("--n_estimators", required=False, default=100, type=int, help='Defined the number of estimators used for the training')
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float, help='Defined the learning rate while training')
    parser.add_argument('--outcome_label', type=str, help='Name of the column with the outcome label')
    parser.add_argument('--output_model', type=str, help='Path of the trained model folder')
    
    args = parser.parse_args()
    print('Printing received arguments...')
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")

    train_model_paynet(args.training_data, args.outcome_label, args.output_model, args.n_estimators, args.learning_rate)
