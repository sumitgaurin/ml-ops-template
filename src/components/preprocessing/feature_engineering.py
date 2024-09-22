import argparse
from genericpath import isfile
import glob
import os
import mlflow
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def feature_engineering(dataset_path:str, output_path:str):
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # Load the complete csv data in a single dataset
        # If the dataset_path is a folder then load all the files
        # else load the data from the single file
        if os.path.isdir(dataset_path):
            # Load the training data from the CSV files
            print('Loacating training dataset files...')
            csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))            
        else:
            csv_files = [dataset_path]
        print(f'Found {len(csv_files)} files in training dataset')
        
        print('Loading training dataset files...')
        df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        print(f'Loaded files in dataframe with schema:')
        print(df.info())
        
        ########################################################################
        # Placeholder for developer to write feature engineering based on the actual implementation
        # Example: 
        scaler = MinMaxScaler()

        transformed_df = pd.DataFrame({
            'Pregnancies': df['Pregnancies'],     
            'Glucose': scaler.fit_transform(df[['Glucose']]).flatten(),
            'BloodPressure': scaler.fit_transform(df[['BloodPressure']]).flatten(),
            'SkinThickness': scaler.fit_transform(df[['SkinThickness']]).flatten(),
            'Insulin': scaler.fit_transform(df[['Insulin']]).flatten(),
            'BMI': scaler.fit_transform(df[['BMI']]).flatten(),
            'DiabetesPedigreeFunction': df['DiabetesPedigreeFunction'],
            'Age': df['Age'],
            'Outcome': df['Outcome']
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
    parser.add_argument('--dataset_path', type=str, help='Path for the input raw data. Can be file or folder')
    parser.add_argument('--output_path', type=str, help='Folder path to save the transformed dataset')
    
    args = parser.parse_args()
    print('Printing received arguments...')
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")
    feature_engineering(args.dataset_path, args.output_path)
