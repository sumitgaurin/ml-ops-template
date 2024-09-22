import pytest
import json
import os
import pandas as pd
from unittest import mock
from src.components.classification.model_selector import compare_models

@pytest.fixture
def mock_metrics_files(tmp_path):
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

    model1_path = tmp_path / "model1_metrics.json"
    model2_path = tmp_path / "model2_metrics.json"

    with open(model1_path, 'w') as f:
        json.dump(model1_metrics, f)
    with open(model2_path, 'w') as f:
        json.dump(model2_metrics, f)

    return [str(model1_path), str(model2_path)]

@pytest.fixture
def output_path(tmp_path):
    return str(tmp_path / "comparison_report.json")

def test_compare_models_minimize_fp(mock_metrics_files, output_path):
    compare_models(mock_metrics_files, 'minimize_fp', output_path)
    
    with open(output_path, 'r') as f:
        result = json.load(f)
    
    assert result['best_model_id'] == 'model_1'
    assert len(result['models']) == 2

def test_compare_models_minimize_fn(mock_metrics_files, output_path):
    compare_models(mock_metrics_files, 'minimize_fn', output_path)
    
    with open(output_path, 'r') as f:
        result = json.load(f)
    
    assert result['best_model_id'] == 'model_2'
    assert len(result['models']) == 2

def test_compare_models_maximize_f1(mock_metrics_files, output_path):
    compare_models(mock_metrics_files, 'maximize_f1', output_path)
    
    with open(output_path, 'r') as f:
        result = json.load(f)
    
    assert result['best_model_id'] == 'model_2'
    assert len(result['models']) == 2