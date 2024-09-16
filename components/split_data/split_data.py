import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_data_path, train_output_dir, test_output_dir, split_ratio=0.7):
    # Load the dataset
    df = pd.read_csv(input_data_path)

    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=(1 - split_ratio), random_state=42)

    # Create directories if they don't exist
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Save the datasets
    train_path = os.path.join(train_output_dir, 'train_data.csv')
    test_path = os.path.join(test_output_dir, 'test_data.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train dataset saved to {train_path}")
    print(f"Test dataset saved to {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='Path to the input dataset')
    parser.add_argument('--train_output', type=str, help='Directory to save the training dataset')
    parser.add_argument('--test_output', type=str, help='Directory to save the testing dataset')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='Train-test split ratio, default is 0.7')

    args = parser.parse_args()
    split_dataset(args.input_data, args.train_output, args.test_output, args.split_ratio)
