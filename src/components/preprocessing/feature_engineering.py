import argparse
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from azureml.core import Dataset, Run

def feature_engineering(dataset_name, output_path):
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
    label_encoder = LabelEncoder()

    transformed_df = pd.DataFrame({
        'feature1_scaled': scaler.fit_transform(df[['feature1']]).flatten(),
        'feature2_encoded': label_encoder.fit_transform(df['feature2'])
    })

    # print the complete transformed DataFrame
    print(transformed_df)    
    ########################################################################
    ########################################################################

    # Save the transformed dataset to the output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    transformed_df.to_csv(output_path, index=False)

    print(f"Transformed dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Name of the input dataset in Azure ML')
    parser.add_argument('--output_path', type=str, help='File path to save the transformed dataset')
    args = parser.parse_args()

    feature_engineering(args.dataset_name, args.output_path)
