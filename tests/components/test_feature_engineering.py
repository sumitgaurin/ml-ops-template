import unittest
from unittest import mock
import pandas as pd
import os
import shutil
from src.components.preprocessing.feature_engineering import feature_engineering

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_output_dir = 'test_output'
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        # Remove the temporary directory after tests
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    @mock.patch('src.components.preprocessing.feature_engineering.mlflow.start_run')
    @mock.patch('src.components.preprocessing.feature_engineering.pd.read_csv')
    @mock.patch('src.components.preprocessing.feature_engineering.glob.glob')
    def test_feature_engineering_single_file(self, mock_glob, mock_read_csv, mock_mlflow):
        # Mock the glob to return a single file
        mock_glob.return_value = ['test_file.csv']
        
        # Create a mock dataframe
        mock_df = pd.DataFrame({
            'Pregnancies': [1, 2],
            'Glucose': [85, 89],
            'BloodPressure': [66, 70],
            'SkinThickness': [29, 30],
            'Insulin': [0, 125],
            'BMI': [26.6, 28.1],
            'DiabetesPedigreeFunction': [0.351, 0.672],
            'Age': [31, 32],
            'Outcome': [0, 1]
        })
        mock_read_csv.return_value = mock_df

        # Call the feature engineering function
        feature_engineering('test_dataset_path', self.test_output_dir)

        # Check if the transformed file is created
        transformed_file_path = os.path.join(self.test_output_dir, 'transformed_data.csv')
        self.assertTrue(os.path.isfile(transformed_file_path))

        # Load the transformed file and check its contents
        transformed_df = pd.read_csv(transformed_file_path)
        self.assertEqual(transformed_df.shape, (2, 9))  # 2 rows, 9 columns

        # assert if mock_mlflow.start_run is called one or more times
        self.assertTrue(mock_mlflow.called)

    @mock.patch('src.components.preprocessing.feature_engineering.mlflow.start_run')
    @mock.patch('src.components.preprocessing.feature_engineering.pd.read_csv')
    @mock.patch('src.components.preprocessing.feature_engineering.glob.glob')
    def test_feature_engineering_multiple_files(self, mock_glob, mock_read_csv, mock_mlflow):
        # Mock the glob to return multiple files
        mock_glob.return_value = ['test_file1.csv', 'test_file2.csv']
        
        # Create mock dataframes
        mock_df1 = pd.DataFrame({
            'Pregnancies': [1],
            'Glucose': [85],
            'BloodPressure': [66],
            'SkinThickness': [29],
            'Insulin': [0],
            'BMI': [26.6],
            'DiabetesPedigreeFunction': [0.351],
            'Age': [31],
            'Outcome': [0]
        })
        mock_df2 = pd.DataFrame({
            'Pregnancies': [2],
            'Glucose': [89],
            'BloodPressure': [70],
            'SkinThickness': [30],
            'Insulin': [125],
            'BMI': [28.1],
            'DiabetesPedigreeFunction': [0.672],
            'Age': [32],
            'Outcome': [1]
        })
        mock_read_csv.side_effect = [mock_df1, mock_df2]

        # Call the feature engineering function
        feature_engineering('test_dataset_path', self.test_output_dir)

        # Check if the transformed file is created
        transformed_file_path = os.path.join(self.test_output_dir, 'transformed_data.csv')
        self.assertTrue(os.path.isfile(transformed_file_path))

        # Load the transformed file and check its contents
        transformed_df = pd.read_csv(transformed_file_path)
        self.assertEqual(transformed_df.shape, (1, 9))  # 2 rows, 9 columns

        # assert if mock_mlflow.start_run is called one or more times
        self.assertTrue(mock_mlflow.called)

if __name__ == '__main__':
    unittest.main()

