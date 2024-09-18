import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from azure.core.exceptions import ResourceNotFoundError
from src.scripts.register_ml_service_assets import asset_exists
from unittest.mock import MagicMock, patch, call
from src.scripts.register_ml_service_assets import register_environments, asset_exists, register_components

def test_asset_exists_true():
    # Mock the asset collection and its get method
    mock_asset_collection = MagicMock()
    mock_asset_collection.get.return_value = True

    # Call the function with the mock
    result = asset_exists(mock_asset_collection, "test_asset", "1.0")

    # Assert the result is True
    assert result == True
    mock_asset_collection.get.assert_called_once_with(name="test_asset", version="1.0")

def test_asset_exists_false():
    # Mock the asset collection and its get method to raise ResourceNotFoundError
    mock_asset_collection = MagicMock()
    mock_asset_collection.get.side_effect = ResourceNotFoundError

    # Call the function with the mock
    result = asset_exists(mock_asset_collection, "test_asset", "1.0")

    # Assert the result is False
    assert result == False
    mock_asset_collection.get.assert_called_once_with(name="test_asset", version="1.0")

def test_asset_exists_exception():
    # Mock the asset collection and its get method to raise a generic exception
    mock_asset_collection = MagicMock()
    mock_asset_collection.get.side_effect = Exception("Unexpected error")

    # Call the function with the mock
    result = asset_exists(mock_asset_collection, "test_asset", "1.0")

    # Assert the result is False
    assert result == False
    mock_asset_collection.get.assert_called_once_with(name="test_asset", version="1.0")

@patch("src.scripts.register_ml_service_assets.os.path.isdir")
@patch("src.scripts.register_ml_service_assets.os.path.isfile")
@patch("src.scripts.register_ml_service_assets.os.listdir")
@patch("src.scripts.register_ml_service_assets.load_environment")
@patch("src.scripts.register_ml_service_assets.asset_exists")
def test_register_environments(asset_exists_mock, load_environment_mock, listdir_mock, isfile_mock, isdir_mock):
    # Mock the inputs
    ml_client_mock = MagicMock()
    src_path = os.path.join(".", "fake", "path")
    ignore_list = ["ignore_env"]

    # Mock the os functions
    isdir_mock.return_value = True
    isfile_mock.return_value = True
    listdir_mock.return_value = ["env1", "env2", "ignore_env"]

    # Mock the load_environment function
    env_asset_mock = MagicMock()
    env_asset_mock.name = "env1"
    env_asset_mock.version = "1.0"
    load_environment_mock.return_value = env_asset_mock

    # Mock the asset_exists function
    asset_exists_mock.return_value = False

    # Call the function
    register_environments(ml_client_mock, src_path, ignore_list)

    # Assertions
    listdir_mock.assert_called_once_with(os.path.join(src_path, "environments"))
    isdir_mock.assert_any_call(os.path.join(src_path, "environments", "env1"))
    isdir_mock.assert_any_call(os.path.join(src_path, "environments", "env2"))
    isfile_mock.assert_any_call(os.path.join(src_path, "environments", "env1", "definition.yaml"))
    isfile_mock.assert_any_call(os.path.join(src_path, "environments", "env2", "definition.yaml"))
    load_environment_mock.assert_any_call(source=os.path.join(src_path, "environments", "env1", "definition.yaml"))
    load_environment_mock.assert_any_call(source=os.path.join(src_path, "environments", "env2", "definition.yaml"))
    asset_exists_mock.assert_any_call(ml_client_mock.environments, "env1", "1.0")
    ml_client_mock.environments.create_or_update.assert_any_call(env_asset_mock)

@patch("src.scripts.register_ml_service_assets.os.path.isdir")
@patch("src.scripts.register_ml_service_assets.os.path.isfile")
@patch("src.scripts.register_ml_service_assets.os.listdir")
@patch("src.scripts.register_ml_service_assets.load_environment")
@patch("src.scripts.register_ml_service_assets.asset_exists")
def test_register_environments_ignore(asset_exists_mock, load_environment_mock, listdir_mock, isfile_mock, isdir_mock):
    # Mock the inputs
    ml_client_mock = MagicMock()
    src_path = os.path.join(".", "fake", "path")
    ignore_list = ["ignore_env"]

    # Mock the os functions
    isdir_mock.return_value = True
    isfile_mock.return_value = True
    listdir_mock.return_value = ["env1", "ignore_env"]

    # Mock the load_environment function
    env_asset_mock = MagicMock()
    env_asset_mock.name = "env1"
    env_asset_mock.version = "1.0"
    load_environment_mock.return_value = env_asset_mock

    # Mock the asset_exists function
    asset_exists_mock.return_value = False

    # Call the function
    register_environments(ml_client_mock, src_path, ignore_list)

    # Assertions
    listdir_mock.assert_called_once_with(os.path.join(src_path, "environments"))
    isdir_mock.assert_any_call(os.path.join(src_path, "environments", "env1"))
    isfile_mock.assert_any_call(os.path.join(src_path, "environments", "env1", "definition.yaml"))
    load_environment_mock.assert_any_call(source=os.path.join(src_path, "environments", "env1", "definition.yaml"))
    asset_exists_mock.assert_any_call(ml_client_mock.environments, "env1", "1.0")
    ml_client_mock.environments.create_or_update.assert_any_call(env_asset_mock)

@patch("src.scripts.register_ml_service_assets.os.path.isdir")
@patch("src.scripts.register_ml_service_assets.os.path.isfile")
@patch("src.scripts.register_ml_service_assets.os.listdir")
def test_register_environments_file_not_found(listdir_mock, isfile_mock, isdir_mock):
    # Mock the inputs
    ml_client_mock = MagicMock()
    src_path = os.path.join(".", "fake", "path")
    ignore_list = []

    # Mock the os functions
    isdir_mock.return_value = True
    isfile_mock.return_value = False
    listdir_mock.return_value = ["env1"]

    # Call the function and assert it raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        register_environments(ml_client_mock, src_path, ignore_list)

    # Assertions
    listdir_mock.assert_called_once_with(os.path.join(src_path, "environments"))
    isdir_mock.assert_any_call(os.path.join(src_path, "environments", "env1"))
    isfile_mock.assert_any_call(os.path.join(src_path, "environments", "env1", "definition.yaml"))
    
@patch("src.scripts.register_ml_service_assets.os.walk")
@patch("src.scripts.register_ml_service_assets.os.path.isfile")
@patch("src.scripts.register_ml_service_assets.load_component")
@patch("src.scripts.register_ml_service_assets.asset_exists")
def test_register_components(asset_exists_mock, load_component_mock, isfile_mock, walk_mock):
    # Mock the inputs
    ml_client_mock = MagicMock()
    src_path = os.path.join(".", "fake", "path")
    ignore_list = ["ignore_component.yaml"]

    # Mock the os functions
    walk_mock.return_value = [
        (os.path.join(src_path, "components"), ["subdir"], ["comp1.yaml", "ignore_component.yaml"]),
        (os.path.join(src_path, "components", "subdir"), [], ["comp2.yaml"])
    ]
    isfile_mock.return_value = True

    # Mock the load_component function
    comp_asset_mock = MagicMock()
    comp_asset_mock.name = "comp1"
    comp_asset_mock.version = "1.0"
    load_component_mock.return_value = comp_asset_mock

    # Mock the asset_exists function
    asset_exists_mock.return_value = False

    # Call the function
    register_components(ml_client_mock, src_path, ignore_list)

    # Assertions
    walk_mock.assert_called_once_with(os.path.join(src_path, "components"))
    load_component_mock.assert_any_call(source=os.path.join(src_path, "components", "comp1.yaml"))
    load_component_mock.assert_any_call(source=os.path.join(src_path, "components", "subdir", "comp2.yaml"))
    asset_exists_mock.assert_any_call(ml_client_mock.components, "comp1", "1.0")
    ml_client_mock.components.create_or_update.assert_any_call(comp_asset_mock)

@patch("src.scripts.register_ml_service_assets.os.walk")
@patch("src.scripts.register_ml_service_assets.os.path.isfile")
@patch("src.scripts.register_ml_service_assets.load_component")
@patch("src.scripts.register_ml_service_assets.asset_exists")
def test_register_components_ignore(asset_exists_mock, load_component_mock, isfile_mock, walk_mock):
    # Mock the inputs
    ml_client_mock = MagicMock()
    src_path = os.path.join(".", "fake", "path")
    ignore_list = ["ignore_component.yaml"]

    # Mock the os functions
    walk_mock.return_value = [
        (os.path.join(src_path, "components"), ["subdir"], ["ignore_component.yaml"])
    ]
    isfile_mock.return_value = True

    # Call the function
    register_components(ml_client_mock, src_path, ignore_list)

    # Assertions
    walk_mock.assert_called_once_with(os.path.join(src_path, "components"))
    load_component_mock.assert_not_called()
    asset_exists_mock.assert_not_called()
    ml_client_mock.components.create_or_update.assert_not_called()


