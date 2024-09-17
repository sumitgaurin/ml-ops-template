import unittest
import os
import json
import pandas as pd
from unittest import mock
from src.components.classification_model_eval import evaluate_model

class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        # Create a temporary test dataset
        self.test_data_path = 'test_data.csv'
        self.output_dir = 'test_output'
        self.outcome_label = 'label'
        df = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3, 0.4],
            'feature2': [1.1, 1.2, 1.3, 1.4],
            self.outcome_label: [0, 1, 0, 1]
        })
        df.to_csv(self.test_data_path, index=False)

        # Prepare mock values for the model
        self.mock_trained_model = mock.Mock()
        self.mock_trained_model.predict.return_value = [0, 1, 0, 1]  # Mock predictions

        # Mock joblib.load to return the mock trained model
        self.joblib_patch = mock.patch('joblib.load', return_value=self.mock_trained_model)
        self.joblib_mock = self.joblib_patch.start()

        # Mock the Model class at the module level in evaluate_model.py
        self.model_patch = mock.patch('src.components.classification_model_eval.Model')
        self.mock_model_class = self.model_patch.start()

        # Create a mock model instance
        self.mock_model_instance = mock.Mock()
        self.mock_model_instance.download.return_value = 'mock_model_path.pkl'
        self.mock_model_instance.name = 'mock_model'

        # When Model is instantiated, return the mock model instance
        self.mock_model_class.return_value = self.mock_model_instance

        # Mock Run.get_context()
        self.run_context_patch = mock.patch('azureml.core.Run.get_context')
        self.mock_run_context = self.run_context_patch.start()
        self.mock_run = mock.Mock()
        self.mock_run.experiment.workspace = mock.Mock()
        self.mock_run_context.return_value = self.mock_run

    def test_evaluate_model(self):
        # Call the evaluate_model function with the mock data
        model_name = 'mock_model'
        model_version = '1'
        results_file_path = os.path.join(self.output_dir, 'metrics.json')
        evaluate_model(model_name, model_version, self.test_data_path, self.outcome_label, results_file_path)

        # Verify that Model was instantiated with the correct parameters
        self.mock_model_class.assert_called_once_with(
            self.mock_run.experiment.workspace, name=model_name, version=model_version
        )

        # Verify that the model's download method was called
        self.mock_model_instance.download.assert_called_once_with(exist_ok=True)

        # Verify that joblib.load was called to load the model
        self.joblib_mock.assert_called_once_with('mock_model_path.pkl')

        # Verify that the trained model's predict method was called with the correct data
        expected_X_test = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3, 0.4],
            'feature2': [1.1, 1.2, 1.3, 1.4]
        })
        actual_X_test = self.mock_trained_model.predict.call_args[0][0]
        pd.testing.assert_frame_equal(expected_X_test, actual_X_test)

        # Verify that the metrics file was created with the correct accuracy
        expected_accuracy = 1.0  # Since we mocked predictions to match labels        
        with open(results_file_path, 'r') as f:
            metrics = json.load(f)
        self.assertEqual(metrics['accuracy'], expected_accuracy)
        self.assertEqual(metrics['recall'], expected_accuracy)
        self.assertEqual(metrics['f1_score'], expected_accuracy)
        self.assertEqual(metrics['model_name'], 'mock_model')
        self.assertEqual(metrics['model_version'], '1')

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
        self.model_patch.stop()
        self.run_context_patch.stop()

if __name__ == '__main__':
    unittest.main()