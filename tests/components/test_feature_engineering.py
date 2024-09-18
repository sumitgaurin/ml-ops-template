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

        self.dataset_name = 'test_dataset'
        self.output_dir = 'test_output'

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Read the sample data from the CSV file
        self.sample_data = pd.read_csv(test_data_file)

        # Mock the Dataset class
        self.mock_dataset = mock.Mock()
        self.mock_dataset.to_pandas_dataframe.return_value = self.sample_data

        # Mock the Run class and its methods
        self.mock_run = mock.Mock()
        self.mock_run.experiment.workspace = 'mock_workspace'

        # Patch the Run.get_context() method to return the mock run
        self.run_patcher = mock.patch('src.components.preprocessing.feature_engineering.Run.get_context', return_value=self.mock_run)
        self.run_patcher.start()

        # Patch the Dataset.get_by_name() method to return the mock dataset
        self.dataset_patcher = mock.patch('src.components.preprocessing.feature_engineering.Dataset.get_by_name', return_value=self.mock_dataset)
        self.dataset_patcher.start()

    def test_feature_engineering(self):
        # Call the feature_engineering function
        output_path = os.path.join(self.output_dir, 'transformed_data.csv')
        feature_engineering(self.dataset_name, output_path)

        # Read the output transformed data        
        self.assertTrue(os.path.exists(output_path), "Transformed data file was not created.")

        transformed_df = pd.read_csv(output_path)

        ########################################################################
        # Placeholder for developer to write assertions based on the actual implementation
        # Example:
        expected_df = pd.DataFrame({
            'feature1_scaled': [0.000000, 0.333333, 0.666667, 1.000000],
            'feature2_encoded': [0, 1, 2, 1]
        })
        pd.testing.assert_frame_equal(transformed_df.reset_index(drop=True), expected_df)
        ########################################################################

        # Temporary assertion to ensure the DataFrame is not empty
        self.assertFalse(transformed_df.empty, "Transformed DataFrame should not be empty.")

    def tearDown(self):
        # Clean up the output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        # Stop patchers
        self.run_patcher.stop()
        self.dataset_patcher.stop()

if __name__ == '__main__':
    unittest.main()
