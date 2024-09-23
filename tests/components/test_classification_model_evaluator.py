import shutil
from unittest import mock
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
import os
from src.components.classification.model_evaluator import evaluate_model


# Test class for testing evaluate_model function including cleanup code for any files or folders generated during the test execution
class TestEvaluateModel:
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

    @mock.patch("src.components.classification.model_evaluator.mlflow")
    @mock.patch("src.components.classification.model_evaluator.load_model")
    @mock.patch("src.components.classification.model_evaluator.glob.glob")
    @mock.patch("src.components.classification.model_evaluator.pd.read_csv")
    @mock.patch("src.components.classification.model_evaluator.os.makedirs")
    def test_evaluate_model_no_model(
        self,
        mock_os_makedirs,
        mock_pd_read_csv,
        mock_glob,
        mock_load_model,
        mock_mlflow,
    ):
        # Setup mocks
        mock_glob.return_value = ["test1.csv", "test2.csv"]

        # create two dataframes with columns feature1, feature2, and outcome
        df1 = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "outcome": [0, 1]})
        df2 = pd.DataFrame({"feature1": [3, 4], "feature2": [5, 6], "outcome": [1, 0]})
        # configure mock_pd_read_csv to return df1 when called with test1.csv and df2 when called with test2.csv
        mock_pd_read_csv.side_effect = lambda filename: {
            'test1.csv': df1,
            'test2.csv': df2
        }.get(filename, pd.DataFrame())        

        # configure the mock_load_model to return None
        mock_load_model.return_value = None

        # Call the function
        result_file = os.path.join(self.test_dir, "results.json")
        evaluate_model(
            "model_1", "path/to/model", "path/to/test_data", "outcome", result_file
        )

        # Check the results
        with open(result_file, "r") as f:
            results = json.load(f)

        assert results["model_id"] == ""
        assert results["accuracy"] == 0
        assert results["recall"] == 0
        assert results["precision"] == 0
        assert results["f1_score"] == 0
        assert results["fpr"] == 1
        assert results["fnr"] == 1

    @mock.patch("src.components.classification.model_evaluator.mlflow")
    @mock.patch("src.components.classification.model_evaluator.load_model")
    @mock.patch("src.components.classification.model_evaluator.glob.glob")
    @mock.patch("src.components.classification.model_evaluator.pd.read_csv")
    @mock.patch("src.components.classification.model_evaluator.os.makedirs")
    def test_evaluate_model_with_model(
        self,
        mock_os_makedirs,
        mock_pd_read_csv,
        mock_glob,
        mock_load_model,
        mock_mlflow,
    ):
        # Setup mocks
        mock_glob.return_value = ["test1.csv", "test2.csv"]

        # create two dataframes with columns feature1, feature2, and outcome
        df1 = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4], "outcome": [0, 1]})
        df2 = pd.DataFrame({"feature1": [3, 4], "feature2": [5, 6], "outcome": [1, 0]})
        # configure mock_pd_read_csv to return df1 when called with test1.csv and df2 when called with test2.csv
        mock_pd_read_csv.side_effect = lambda filename: {
            'test1.csv': df1,
            'test2.csv': df2
        }.get(filename, pd.DataFrame())
        
        # configure the mock_load_model to return a mock model that predicts [0, 1, 1, 0]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 1, 0]
        mock_load_model.return_value = mock_model

        # Call the function
        result_file = os.path.join(self.test_dir, "results.json")
        evaluate_model(
            "model_1", "path/to/model", "path/to/test_data", "outcome", result_file
        )

        # Check the results
        with open(result_file, "r") as f:
            results = json.load(f)

        assert results["model_id"] == "model_1"
        assert results["accuracy"] == 1.0
        assert results["recall"] == 1.0
        assert results["precision"] == 1.0
        assert results["f1_score"] == 1.0
        assert results["fpr"] == 0.0
        assert results["fnr"] == 0.0
