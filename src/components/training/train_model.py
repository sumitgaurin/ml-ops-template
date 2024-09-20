import argparse
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier

def train_model_paynet(training_data_path, output_model_dir):
    # Start Logging
    mlflow.start_run()
    # enable autologging
    mlflow.sklearn.autolog()

    # Load the training data from the CSV file
    df = pd.read_csv(training_data_path)
    
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
    os.makedirs(output_model_dir, exist_ok=True)
    model_output_path = os.path.join(output_model_dir, 'trained_model.pkl')
    joblib.dump(model, model_output_path)
    
    print(f"Model saved at {model_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Define arguments for the script
    parser.add_argument('--training_data', type=str, help='Path to the training data CSV file')
    parser.add_argument('--output_model', type=str, help='Directory to save the trained model')
    
    args = parser.parse_args()

    # Train the model and save it to the output directory
    train_model_paynet(args.training_data, args.output_model)
