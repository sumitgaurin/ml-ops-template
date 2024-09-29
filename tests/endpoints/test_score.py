import json
import os
import pytest
import joblib
from unittest.mock import patch, MagicMock
from src.endpoints.scoring.score import init, run

# Ensure the mock_model_dir is correctly set
mock_model_dir = os.path.join(".", "azureml-models", "diabetes_model", "1")
mock_model_path = os.path.join(mock_model_dir, "model", "model.pkl")
mock_model = MagicMock()
mock_model.predict_proba.return_value = [[0.3, 0.7]]

# Sample input data
sample_input = json.dumps({
    'Pregnancies': 2,
    'Glucose': 120,
    'BloodPressure': 70,
    'SkinThickness': 20,
    'Insulin': 85,
    'BMI': 25.0,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 30
})

# Expected output data
expected_output = {
    "predicted_class": "Diabetes",
    "probability": 0.7,
    "model_name": "diabetes_model",
    "model_version": "1"
}

@pytest.fixture
def setup_environment(monkeypatch):
    # Mock environment variable
    monkeypatch.setenv("AZUREML_MODEL_DIR", mock_model_dir)
    # Mock joblib.load to return the mock model
    with patch("joblib.load", return_value=mock_model):
        init()

def test_init(setup_environment):
    assert os.environ.get("AZUREML_MODEL_DIR") == mock_model_dir

def test_run_success(setup_environment):
    response = run(sample_input)
    response_data = json.loads(response)
    assert response_data == expected_output

def test_run_missing_field(setup_environment):
    incomplete_input = json.dumps({
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 20,
        'Insulin': 85,
        'BMI': 25.0,
        'DiabetesPedigreeFunction': 0.5
        # Missing 'Age'
    })
    response = run(incomplete_input)
    response_data = json.loads(response)
    assert "error" in response_data
    assert "Missing input field" in response_data["error"]

def test_run_invalid_json(setup_environment):
    invalid_json_input = "{'Pregnancies': 2, 'Glucose': 120}"  # Invalid JSON format
    response = run(invalid_json_input)
    response_data = json.loads(response)
    assert "error" in response_data
    assert "Expecting property name enclosed in double quotes" in response_data["error"]

def test_run_unexpected_field(setup_environment):
    unexpected_field_input = json.dumps({
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 20,
        'Insulin': 85,
        'BMI': 25.0,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 30,
        'UnexpectedField': 999
    })
    response = run(unexpected_field_input)
    response_data = json.loads(response)
    assert response_data == expected_output

def test_run_empty_input(setup_environment):
    empty_input = json.dumps({})
    response = run(empty_input)
    response_data = json.loads(response)
    assert "error" in response_data
    assert "Missing input field" in response_data["error"]

def test_run_non_numeric_input(setup_environment):
    non_numeric_input = json.dumps({
        'Pregnancies': 'two',
        'Glucose': 'one twenty',
        'BloodPressure': 'seventy',
        'SkinThickness': 'twenty',
        'Insulin': 'eighty-five',
        'BMI': 'twenty-five',
        'DiabetesPedigreeFunction': 'zero point five',
        'Age': 'thirty'
    })
    response = run(non_numeric_input)
    response_data = json.loads(response)
    assert "error" in response_data
    assert "could not convert string to float" in response_data["error"]
