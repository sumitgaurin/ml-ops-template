import unittest
import os
import json
import pandas as pd
from unittest import mock
from sklearn.metrics import accuracy_score
from classification_model_eval import evaluate_model

class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        # Create a temporary test dataset
        self.test_data_path = 'test_data.csv'
        self.output_dir = 'output'
        df = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3, 0.4],
            'feature2': [1.1, 1.2, 1.3, 1.4],
            'label': [0, 1, 0, 1]
        })
        df.to_csv(self.test_data_path, index=False)

        # Prepare mock values for model
        self.mock_model = mock.Mock()
        self.mock_model.predict.return_value = [0, 1, 0, 1]  # Mock predictions

        # Mock the joblib.load to return the mock model
        self.joblib_patch = mock.patch('joblib.load', return_value=self.mock_model)
        self.joblib_mock = self.joblib_patch.start()

        # Mock AzureML Model
        self.mock_model_download = mock.Mock(return_value='mock_model_path.pkl')
        self.mock_azure_model = mock.Mock()
        self.mock_azure_model.download = self.mock_model_download

        # Mock AzureML Workspace and Run
        self.mock_ws = mock.Mock()
        self.mock_run = mock.Mock()
        self.mock_run.experiment.workspace = self.mock_ws
        self.mock_run_context_patch = mock.patch('azureml.core.Run.get_context', return_value=self.mock_run)
        self.run_context_mock = self.mock_run_context_patch.start()

        # Mock the AzureML Model class
        self.azureml_model_patch = mock.patch('azureml.core.Model', return_value=self.mock_azure_model)
        self.azureml_model_mock = self.azureml_model_patch.start()

    def test_evaluate_model(self):
        # Call the evaluate_model function with the mock data
        model_name = 'mock_model'
        model_version = '1'
        evaluate_model(model_name, model_version, self.test_data_path, self.output_dir)

        # Verify that joblib.load was called to load the model
        self.joblib_mock.assert_called_once_with('mock_model_path.pkl')

        # Verify that the model's predict method was called with the correct data
        pd.testing.assert_frame_equal(
            pd.DataFrame({'feature1': [0.1, 0.2, 0.3, 0.4], 'feature2': [1.1, 1.2, 1.3, 1.4]}),
            pd.DataFrame(self.mock_model.predict.call_args[0][0])
        )

        # Verify that the accuracy is calculated correctly
        expected_accuracy = accuracy_score([0, 1, 0, 1], [0, 1, 0, 1])
        results_file_path = os.path.join(self.output_dir, 'metrics.json')
        with open(results_file_path, 'r') as f:
            metrics = json.load(f)
        self.assertEqual(metrics['accuracy'], expected_accuracy)

    def tearDown(self):
        # Clean up the test environment
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)

        # Stop all patches
        self.joblib_patch.stop()
        self.run_context_mock.stop()
        self.azureml_model_patch.stop()

if __name__ == '__main__':
    unittest.main()
