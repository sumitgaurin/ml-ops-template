import unittest
import os
import pandas as pd
from unittest import mock
from src.components.training.train_model import train_model_paynet

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset for testing
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'label': [0, 1, 0, 1]
        })

        # Set paths for the test
        self.training_data_path = 'test_training_data.csv'
        self.output_model_dir = 'test_output_model'

        # Save the sample data to a CSV file (mocked later)
        self.sample_data.to_csv(self.training_data_path, index=False)

        # Mock the os.makedirs function to avoid creating directories
        self.makedirs_patcher = mock.patch('os.makedirs')
        self.mock_makedirs = self.makedirs_patcher.start()

        # Mock the joblib.dump function to avoid writing to disk
        self.joblib_dump_patcher = mock.patch('joblib.dump')
        self.mock_joblib_dump = self.joblib_dump_patcher.start()

    def test_train_model(self):
        # Mock the pd.read_csv function to return the sample data
        with mock.patch('pandas.read_csv', return_value=self.sample_data):
            # Call the train_model function
            train_model_paynet(self.training_data_path, self.output_model_dir)

            # Check that the model was saved using joblib
            model_output_path = os.path.join(self.output_model_dir, 'trained_model.pkl')
            self.mock_joblib_dump.assert_called_with(mock.ANY, model_output_path)

    def tearDown(self):
        # Clean up the patchers
        self.makedirs_patcher.stop()
        self.joblib_dump_patcher.stop()

        # Remove the test CSV file
        if os.path.exists(self.training_data_path):
            os.remove(self.training_data_path)

        # Remove the output directory if it was created (mocked, so this may not be necessary)
        if os.path.exists(self.output_model_dir):
            os.rmdir(self.output_model_dir)

if __name__ == '__main__':
    unittest.main()
