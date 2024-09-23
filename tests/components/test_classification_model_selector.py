import shutil
import pytest
import json
import os
import pandas as pd
from unittest import mock
from src.components.classification.model_selector import compare_models

class TestCompareModels:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Setup: Create a temporary directory for the test
        self.test_dir = "test_output"
        os.mkdir(self.test_dir)

        # Teardown: Cleanup any files or folders generated during the test
        yield
        # Cleanup the temporary directory recursively along with all its content using shutil.rmtree
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @pytest.fixture
    def mock_metrics_files(self):
        # Create temporary JSON files with mock metrics
        model1_metrics = {
            "model_id": "model_1",
            "f1_score": 0.8,
            "fpr": 0.1,
            "fnr": 0.2
        }
        model2_metrics = {
            "model_id": "model_2",
            "f1_score": 0.85,
            "fpr": 0.15,
            "fnr": 0.1
        }

        model1_path = os.path.join(self.test_dir,"model1_metrics.json")
        model2_path = os.path.join(self.test_dir,"model2_metrics.json")

        with open(model1_path, 'w') as f:
            json.dump(model1_metrics, f)
        with open(model2_path, 'w') as f:
            json.dump(model2_metrics, f)

        return [str(model1_path), str(model2_path)]

    @pytest.fixture
    def output_path(self):
        return os.path.join(self.test_dir, "comparison_report.json")

    @mock.patch('src.components.classification.model_selector.mlflow')
    def test_compare_models_minimize_fp(self, mock_mlflow, mock_metrics_files, output_path):
        # Call the function to compare models
        compare_models(mock_metrics_files, 'minimize_fp', output_path)
        
        with open(output_path, 'r') as f:
            result = json.load(f)
        
        assert result['best_model_id'] == 'model_1'
        assert len(result['models']) == 2

        # Assert that mlflow start_run and log_metrics were called
        mock_mlflow.start_run.assert_called()

    @mock.patch('src.components.classification.model_selector.mlflow')
    def test_compare_models_minimize_fn(self, mock_mlflow, mock_metrics_files, output_path):
        compare_models(mock_metrics_files, 'minimize_fn', output_path)
        
        with open(output_path, 'r') as f:
            result = json.load(f)
        
        assert result['best_model_id'] == 'model_2'
        assert len(result['models']) == 2

        # Assert that mlflow start_run and log_metrics were called
        mock_mlflow.start_run.assert_called()

    @mock.patch('src.components.classification.model_selector.mlflow')
    def test_compare_models_maximize_f1(self, mock_mlflow, mock_metrics_files, output_path):
        compare_models(mock_metrics_files, 'maximize_f1', output_path)
        
        with open(output_path, 'r') as f:
            result = json.load(f)
        
        assert result['best_model_id'] == 'model_2'
        assert len(result['models']) == 2

        # Assert that mlflow start_run and log_metrics were called
        mock_mlflow.start_run.assert_called()