import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
import os
from src.components.classification.model_evaluator import evaluate_model

@pytest.fixture
def mock_mlflow():
    with patch('src.components.classification.model_evaluator.mlflow') as mock_mlflow:
        yield mock_mlflow

@pytest.fixture
def mock_load_model():
    with patch('src.components.classification.model_evaluator.load_model') as mock_load_model:
        yield mock_load_model

@pytest.fixture
def mock_glob():
    with patch('src.components.classification.model_evaluator.glob.glob') as mock_glob:
        yield mock_glob

@pytest.fixture
def mock_pd_read_csv():
    with patch('src.components.classification.model_evaluator.pd.read_csv') as mock_read_csv:
        yield mock_read_csv

@pytest.fixture
def mock_os_makedirs():
    with patch('src.components.classification.model_evaluator.os.makedirs') as mock_makedirs:
        yield mock_makedirs

def test_evaluate_model_no_model(mock_mlflow, mock_load_model, mock_glob, mock_pd_read_csv, mock_os_makedirs):
    # Setup mocks
    mock_glob.return_value = ['test1.csv', 'test2.csv']
    mock_pd_read_csv.side_effect = [pd.DataFrame({'feature1': [1, 2], 'outcome': [0, 1]}),
                                    pd.DataFrame({'feature1': [3, 4], 'outcome': [1, 0]})]
    mock_load_model.return_value = None

    # Call the function
    result_file = 'results.json'
    evaluate_model('model_1', 'path/to/model', 'path/to/test_data', 'outcome', result_file)

    # Check the results
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # cleanup the temp result file
    os.remove(result_file)

    assert results['model_id'] == ""
    assert results['accuracy'] == 0
    assert results['recall'] == 0
    assert results['precision'] == 0
    assert results['f1_score'] == 0
    assert results['fpr'] == 1
    assert results['fnr'] == 1

def test_evaluate_model_with_model(mock_mlflow, mock_load_model, mock_glob, mock_pd_read_csv, mock_os_makedirs):
    # Setup mocks
    mock_glob.return_value = ['test1.csv', 'test2.csv']
    mock_pd_read_csv.side_effect = [pd.DataFrame({'feature1': [1, 2], 'outcome': [0, 1]}),
                                    pd.DataFrame({'feature1': [3, 4], 'outcome': [1, 0]})]
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1, 1, 0]
    mock_load_model.return_value = mock_model

    # Call the function
    result_file = 'results.json'
    evaluate_model('model_1', 'path/to/model', 'path/to/test_data', 'outcome', result_file)

    # Check the results
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # cleanup the temp result file
    os.remove(result_file)
    
    assert results['model_id'] == "model_1"
    assert results['accuracy'] == 1.0
    assert results['recall'] == 1.0
    assert results['precision'] == 1.0
    assert results['f1_score'] == 1.0
    assert results['fpr'] == 0.0
    assert results['fnr'] == 0.0