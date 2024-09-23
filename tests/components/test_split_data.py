import shutil
import unittest
import os
import pandas as pd
from src.components.training.split_data import split_dataset
from unittest.mock import patch

class TestSplitDataset(unittest.TestCase):

    def setUp(self):
        # Setup paths for input and output
        self.test_dir = "test_output_data"
        self.input_data_path = 'test_input_data'
        self.train_output_path = os.path.join(self.test_dir, 'train')
        self.test_output_path = os.path.join(self.test_dir, 'test')
        
        # Create directories
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.input_data_path, exist_ok=True)
        os.makedirs(self.train_output_path, exist_ok=True)
        os.makedirs(self.test_output_path, exist_ok=True)
        
        # Create a sample CSV file
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'label': [0, 1, 0, 1, 0]
        })
        self.sample_data.to_csv(os.path.join(self.input_data_path, 'sample.csv'), index=False)

    def tearDown(self):
        # Cleanup created directories and files
        for path in [self.test_dir, self.input_data_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    @patch('mlflow.start_run')
    @patch('mlflow.sklearn.autolog')
    def test_split_dataset(self, mock_autolog, mock_start_run):
        # Run the split_dataset function
        split_dataset(self.input_data_path, self.train_output_path, self.test_output_path, split_ratio=0.7)
        
        # Check if the train and test files are created
        train_file = os.path.join(self.train_output_path, 'train_data.csv')
        test_file = os.path.join(self.test_output_path, 'test_data.csv')
        
        self.assertTrue(os.path.exists(train_file))
        self.assertTrue(os.path.exists(test_file))
        
        # Load the datasets
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # Check if the split ratio is approximately correct
        total_size = len(self.sample_data)
        train_size = len(train_df)
        test_size = len(test_df)
        
        self.assertAlmostEqual(train_size / total_size, 0.7, delta=0.2)
        self.assertAlmostEqual(test_size / total_size, 0.3, delta=0.2)

        # Assert that the mlflow function is called atleast once
        mock_start_run.assert_called()
        mock_autolog.assert_called()

    @patch('mlflow.start_run')
    @patch('mlflow.sklearn.autolog')
    def test_split_dataset_different_ratio(self, mock_autolog, mock_start_run):
        # Run the split_dataset function with a different split ratio
        split_dataset(self.input_data_path, self.train_output_path, self.test_output_path, split_ratio=0.5)
        
        # Check if the train and test files are created
        train_file = os.path.join(self.train_output_path, 'train_data.csv')
        test_file = os.path.join(self.test_output_path, 'test_data.csv')
        
        self.assertTrue(os.path.exists(train_file))
        self.assertTrue(os.path.exists(test_file))
        
        # Load the datasets
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # Check if the split ratio is approximately correct
        total_size = len(self.sample_data)
        train_size = len(train_df)
        test_size = len(test_df)
        
        self.assertAlmostEqual(train_size / total_size, 0.5, delta=0.2)
        self.assertAlmostEqual(test_size / total_size, 0.5, delta=0.2)

        # Assert that the mlflow function is called atleast once
        mock_start_run.assert_called()
        mock_autolog.assert_called()

    @patch('mlflow.start_run')
    @patch('mlflow.sklearn.autolog')
    def test_split_dataset_no_data(self, mock_autolog, mock_start_run):
        # Remove sample data to simulate no data scenario
        os.remove(os.path.join(self.input_data_path, 'sample.csv'))
        
        with self.assertRaises(ValueError):
            split_dataset(self.input_data_path, self.train_output_path, self.test_output_path, split_ratio=0.7)
        
        # Assert that the mlflow function is called atleast once
        mock_start_run.assert_called()
        mock_autolog.assert_called()
        

if __name__ == '__main__':
    unittest.main()
