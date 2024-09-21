import argparse
import glob
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from src.components.helper import print_args


def split_dataset(input_data_path, train_path, test_path, split_ratio=0.7):
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # enable autologging
        mlflow.sklearn.autolog()

        # Load the training data from the CSV files
        print('Loacating training feature dataset files...')
        csv_files = glob.glob(os.path.join(input_data_path, '*.csv'))
        print(f'Found {csv_files.count} files in training feature dataset')

        print('Loading training feature dataset files...')
        df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        print(f'Loaded files in dataframe with schema:')
        print(df.info())

        # Split the dataset
        train_df, test_df = train_test_split(df, test_size=(1 - split_ratio), random_state=42)

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train dataset saved to {train_path}")
        print(f"Test dataset saved to {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='Path to the input dataset')
    parser.add_argument('--train_output', type=str, help='File path to save the training dataset')
    parser.add_argument('--test_output', type=str, help='File path to save the testing dataset')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='Train-test split ratio, default is 0.7')

    args = parser.parse_args()
    print_args(args)
    split_dataset(args.input_data, args.train_output, args.test_output, args.split_ratio)
