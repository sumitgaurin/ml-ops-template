import argparse
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from azureml.core import Dataset, Run

def feature_engineering(dataset_name, output_dir):
    # Get the workspace and dataset
    run = Run.get_context()
    ws = run.experiment.workspace
    dataset = Dataset.get_by_name(ws, name=dataset_name)

    # Load the dataset into a pandas DataFrame
    df = dataset.to_pandas_dataframe()
    
    ########################################################################
    ########################################################################
    # Write your code here to result in transformed_df
    transformed_df = df
    ########################################################################
    ########################################################################

    # Save the transformed dataset to the output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'transformed_data.csv')
    transformed_df.to_csv(output_path, index=False)

    print(f"Transformed dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Name of the input dataset in Azure ML')
    parser.add_argument('--output_dir', type=str, help='Directory to save the transformed dataset')
    args = parser.parse_args()

    feature_engineering(args.dataset_name, args.output_dir)
