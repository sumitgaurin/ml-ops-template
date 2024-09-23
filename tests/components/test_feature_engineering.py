import unittest
from unittest import mock
import pandas as pd
import os
import shutil
from src.components.preprocessing.feature_engineering import feature_engineering

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Compute the path to the test_data.csv file relative to this script
        current_dir = os.path.dirname(__file__)
        test_data_file = os.path.join(current_dir, '..', 'data', 'test_data.csv')

        self.dataset_path = test_data_file
        self.output_dir = 'test_output'

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Read the sample data from the CSV file
        self.sample_data = pd.read_csv(test_data_file)

        # Mock mlflow
        self.mlflow_patcher = mock.patch('src.components.preprocessing.feature_engineering.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()

    def test_feature_engineering(self):
        # Call the feature_engineering function
        feature_engineering(self.dataset_path, self.output_dir)

        # Read the output transformed data        
        output_path = os.path.join(self.output_dir, 'transformed_data.csv')
        self.assertTrue(os.path.exists(output_path), "Transformed data file was not created.")

        transformed_df = pd.read_csv(output_path)

        # Check if the transformed DataFrame has the expected columns
        expected_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        self.assertListEqual(list(transformed_df.columns), expected_columns, "Transformed DataFrame columns do not match expected columns.")

        # Check if the transformed DataFrame is not empty
        self.assertFalse(transformed_df.empty, "Transformed DataFrame should not be empty.")

        # Check if mlflow metrics were logged
        self.mock_mlflow.log_metric.assert_any_call("num_samples", transformed_df.shape[0])
        self.mock_mlflow.log_metric.assert_any_call("num_features", len(expected_columns) - 1)

    def tearDown(self):
        # Clean up the output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        # Stop patchers
        self.mlflow_patcher.stop()

if __name__ == '__main__':
    unittest.main()
