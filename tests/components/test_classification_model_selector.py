import shutil
import unittest
import pytest
import json
import os
import pandas as pd
from unittest import mock
from src.components.classification.model_selector import compare_models

class TestCompareModels(unittest.TestCase):

    def setUp(self):
        # Setup: Create a temporary directory for the test
        self.test_dir = "test_output"
        os.mkdir(self.test_dir)

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

        self.mock_metrics_files = [str(model1_path), str(model2_path)]
        self.output_path = os.path.join(self.test_dir, "comparison_report.json")

    def tearDown(self):
        # Cleanup created directories and files
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @mock.patch('src.components.classification.model_selector.mlflow')
    def test_compare_models_minimize_fp(self, mock_mlflow):
        # Call the function to compare models
        compare_models(self.mock_metrics_files, 'minimize_fp', self.output_path)
        
        with open(self.output_path, 'r') as f:
            result = json.load(f)
        
        assert result['best_model_id'] == 'model_1'
        assert len(result['models']) == 2

        # Assert that mlflow start_run and log_metrics were called
        mock_mlflow.start_run.assert_called()

    @mock.patch('src.components.classification.model_selector.mlflow')
    def test_compare_models_minimize_fn(self, mock_mlflow):
        compare_models(self.mock_metrics_files, 'minimize_fn', self.output_path)
        
        with open(self.output_path, 'r') as f:
            result = json.load(f)
        
        assert result['best_model_id'] == 'model_2'
        assert len(result['models']) == 2

        # Assert that mlflow start_run and log_metrics were called
        mock_mlflow.start_run.assert_called()

    @mock.patch('src.components.classification.model_selector.mlflow')
    def test_compare_models_maximize_f1(self, mock_mlflow):
        compare_models(self.mock_metrics_files, 'maximize_f1', self.output_path)
        
        with open(self.output_path, 'r') as f:
            result = json.load(f)
        
        assert result['best_model_id'] == 'model_2'
        assert len(result['models']) == 2

        # Assert that mlflow start_run and log_metrics were called
        mock_mlflow.start_run.assert_called()