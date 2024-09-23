import json
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import mlflow

from src.components.training.register_model import register_trained_model

class TestRegisterTrainedModel(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_output"
        os.makedirs(self.test_dir, exist_ok=True)
        self.model_path = os.path.join(self.test_dir, "model")
        os.makedirs(self.model_path, exist_ok=True)

        self.comparison_report_path = os.path.join(self.test_dir, "comparison_report.json")
        self.register_report_path = os.path.join(self.test_dir, "register_report.txt")
        self.model_name = "test_model"
        self.model_id = "trained"

        # Create a dummy comparison report
        with open(self.comparison_report_path, "w") as f:
            json.dump({"best_model_id": self.model_id}, f)

        # Create a dummy model file
        with open(os.path.join(self.model_path, 'model.pkl'), "w") as f:
            f.write("dummy model content")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("mlflow.start_run")
    @patch('mlflow.sklearn.autolog')
    @patch("mlflow.sklearn.load_model")
    @patch("mlflow.sklearn.log_model")
    def test_register_trained_model_success(self, mock_log_model, mock_load_model, mock_autolog, mock_start_run):
        # Mock the return value of mlflow.sklearn.load_model
        model = MagicMock()
        mock_load_model.return_value = model

        # Call the function to register the trained model
        register_trained_model(
            self.comparison_report_path,
            self.model_path,
            self.model_name,
            self.model_id,
            self.register_report_path,
        )

        # Assert that mlflow functions were called correctly
        mock_start_run.assert_called()
        mock_autolog.assert_called()
        mock_load_model.assert_called_once_with(self.model_path)
        mock_log_model.assert_called_once_with(
            sk_model=model,
            registered_model_name=self.model_name,
            artifact_path=self.model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        with open(self.register_report_path, "r") as f:
            report_content = f.read()
            self.assertIn("Model registration complete.", report_content)

    @patch("mlflow.start_run")
    @patch('mlflow.sklearn.autolog')
    def test_register_trained_model_comparison_report_not_found(self, mock_autolog, mock_start_run):
        # Create a dummy path to a non-existent comparison report
        invalid_comparison_report_path = os.path.join(self.test_dir, "invalid_comparison_report.json")

        # Call the function to register the trained model
        register_trained_model(
            invalid_comparison_report_path,
            self.model_path,
            self.model_name,
            self.model_id,
            self.register_report_path,
        )

        # Assert that mlflow functions were called correctly
        mock_start_run.assert_called()
        mock_autolog.assert_called()
        with open(self.register_report_path, "r") as f:
            report_content = f.read()
            self.assertIn("Comparison report not found.", report_content)
            self.assertIn("Model registration aborted.", report_content)

    @patch("mlflow.start_run")
    @patch('mlflow.sklearn.autolog')
    def test_register_trained_model_existing_model_better(self, mock_autolog, mock_start_run):
        # Create a comparison report where the existing model is better
        with open(self.comparison_report_path, "w") as f:
            json.dump({"best_model_id": "different_model_id"}, f)
        
        # Call the function to register the trained model
        register_trained_model(
            self.comparison_report_path,
            self.model_path,
            self.model_name,
            self.model_id,
            self.register_report_path,
        )

        # Assertions
        mock_start_run.assert_called()
        mock_autolog.assert_called()
        with open(self.register_report_path, "r") as f:
            report_content = f.read()
            self.assertIn("Existing model is better than trained model.", report_content)
            self.assertIn("Model registration aborted.", report_content)

if __name__ == "__main__":
    unittest.main()
