import argparse
import os
import mlflow
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from azureml.core import Dataset, Run

def feature_engineering(dataset_name, output_path):
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # Get the workspace and dataset
        run = Run.get_context()
        ws = run.experiment.workspace
        dataset = Dataset.get_by_name(ws, name=dataset_name)

        # Load the dataset into a pandas DataFrame
        df = dataset.to_pandas_dataframe()
        
        ########################################################################
        # Placeholder for developer to write feature engineering based on the actual implementation
        # Example: 
        scaler = MinMaxScaler()

        transformed_df = pd.DataFrame({
            'Pregnancies': df[['Pregnancies']],
            'Glucose': scaler.fit_transform(df[['Glucose']]).flatten(),
            'BloodPressure': scaler.fit_transform(df[['BloodPressure']]).flatten(),
            'SkinThickness': scaler.fit_transform(df[['SkinThickness']]).flatten(),
            'Insulin': scaler.fit_transform(df[['Insulin']]).flatten(),
            'BMI': scaler.fit_transform(df[['BMI']]).flatten(),
            'DiabetesPedigreeFunction': df[['DiabetesPedigreeFunction']],
            'Age': df[['Pregnancies']],
            'Outcome': df[['Outcome']]
        })

        # log mlflow metric
        mlflow.log_metric("num_samples", transformed_df.shape[0])
        mlflow.log_metric("num_features", transformed_df.shape[1] - 1)

        # print the sample of transformed DataFrame
        print(transformed_df.head(3))    
        ########################################################################
        ########################################################################

        # Save the transformed dataset to the output directory
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, "transformed_data.csv")
        transformed_df.to_csv(file_path, index=False)

        print(f"Transformed dataset saved to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Name of the input dataset in Azure ML')
    parser.add_argument('--output_path', type=str, help='File path to save the transformed dataset')
    
    args = parser.parse_args()
    print('Printing received arguments...')
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")
    feature_engineering(args.dataset_name, args.output_path)
