import pytest
from unittest.mock import patch, MagicMock
from src.components.training.register_model import register_model

@patch("src.components.training.register_model.Run.get_context")
@patch("src.components.training.register_model.Model.register")
def test_register_model(mock_model_register, mock_run_get_context):
    # Mock the workspace and model registration
    mock_workspace = MagicMock()
    mock_run = MagicMock()
    mock_run.experiment.workspace = mock_workspace
    mock_run_get_context.return_value = mock_run

    mock_model = MagicMock()
    mock_model.name = "test_model"
    mock_model.version = "1"
    mock_model_register.return_value = mock_model

    # Call the function with test parameters
    model_name = "test_model"
    model_path = "test_path"
    registered_model = ""
    register_model(model_name, model_path, registered_model)

    # Assertions to ensure the function behaves as expected
    mock_run_get_context.assert_called_once()
    mock_model_register.assert_called_once_with(workspace=mock_workspace, model_name=model_name, model_path=model_path)
    assert mock_model.name == "test_model"
    assert mock_model.version == "1"

@patch("src.components.training.register_model.Run.get_context")
@patch("src.components.training.register_model.Model.register")
def test_register_model_output(mock_model_register, mock_run_get_context):
    # Mock the workspace and model registration
    mock_workspace = MagicMock()
    mock_run = MagicMock()
    mock_run.experiment.workspace = mock_workspace
    mock_run_get_context.return_value = mock_run

    mock_model = MagicMock()
    mock_model.name = "test_model"
    mock_model.version = "1"
    mock_model_register.return_value = mock_model

    # Call the function with test parameters
    model_name = "test_model"
    model_path = "test_path"
    registered_model = "test_model:1"
    register_model(model_name, model_path, registered_model)

    # Check the registered_model output
    expected_registered_model = f"{mock_model.name}:{mock_model.version}"
    assert registered_model == expected_registered_model
