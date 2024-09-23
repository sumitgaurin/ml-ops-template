import argparse
import glob
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

def split_dataset(input_data_path:str, train_path:str, test_path:str, split_ratio:float=0.7)->None:
    """
    Splits a dataset into training and testing sets and saves them to specified paths.

    Parameters
    ----------
    input_data_path : str
        The directory path where the input CSV files are located.
    
    train_path : str 
        The directory path where the training dataset will be saved.
        
    test_path : str
        The directory path where the testing dataset will be saved.

    split_ratio : float, optional
        The ratio of the dataset to be used for training. 
        The default value is 0.7, meaning 70% of the data will be used for training and 30% for testing.
        
    Returns
    -------
    None : The function saves the training and testing datasets to the specified paths.
    """
    # Start Logging with mlflow using context manager
    with mlflow.start_run():
        # enable autologging
        mlflow.sklearn.autolog()

        # Load the training data from the CSV files
        print('Loacating training feature dataset files...')
        csv_files = glob.glob(os.path.join(input_data_path, '*.csv'))
        print(f'Found {len(csv_files)} files in training feature dataset')

        print('Loading training feature dataset files...')
        df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        print(f'Loaded files in dataframe with schema:')
        print(df.info())

        # Split the dataset
        train_df, test_df = train_test_split(df, test_size=(1 - split_ratio), random_state=42)

        # Create directories if they don't exist
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        train_file = os.path.join(train_path, "train_data.csv")
        train_df.to_csv(train_file, index=False)
        print(f"Train dataset with {train_df.size} saved to {train_path}")

        test_file = os.path.join(test_path, "test_data.csv")
        test_df.to_csv(test_file, index=False)        
        print(f"Test dataset with {test_df.size} saved to {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='Path to the input dataset')
    parser.add_argument('--train_output', type=str, help='File path to save the training dataset')
    parser.add_argument('--test_output', type=str, help='File path to save the testing dataset')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='Train-test split ratio, default is 0.7')

    args = parser.parse_args()
    print('Printing received arguments...')
    for arg_name in vars(args):
        print(f"{arg_name}: {getattr(args, arg_name)}")

    split_dataset(args.input_data, args.train_output, args.test_output, args.split_ratio)
