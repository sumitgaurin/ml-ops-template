import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from azure.core.exceptions import ResourceNotFoundError
from src.scripts.register_ml_service_assets import register_environments, register_components

@patch("src.scripts.register_ml_service_assets.os.path.isdir")
@patch("src.scripts.register_ml_service_assets.os.path.isfile")
@patch("src.scripts.register_ml_service_assets.os.listdir")
@patch("src.scripts.register_ml_service_assets.load_environment")
def test_register_environments(load_environment_mock, listdir_mock, isfile_mock, isdir_mock):
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
    ml_client_mock.environments.create_or_update.assert_any_call(env_asset_mock)

@patch("src.scripts.register_ml_service_assets.os.path.isdir")
@patch("src.scripts.register_ml_service_assets.os.path.isfile")
@patch("src.scripts.register_ml_service_assets.os.listdir")
@patch("src.scripts.register_ml_service_assets.load_environment")
def test_register_environments_ignore(load_environment_mock, listdir_mock, isfile_mock, isdir_mock):
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

    # Call the function
    register_environments(ml_client_mock, src_path, ignore_list)

    # Assertions
    listdir_mock.assert_called_once_with(os.path.join(src_path, "environments"))
    isdir_mock.assert_any_call(os.path.join(src_path, "environments", "env1"))
    isfile_mock.assert_any_call(os.path.join(src_path, "environments", "env1", "definition.yaml"))
    load_environment_mock.assert_any_call(source=os.path.join(src_path, "environments", "env1", "definition.yaml"))
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
def test_register_components(load_component_mock, isfile_mock, walk_mock):
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

    # Call the function
    register_components(ml_client_mock, src_path, ignore_list)

    # Assertions
    walk_mock.assert_called_once_with(os.path.join(src_path, "components"))
    load_component_mock.assert_any_call(source=os.path.join(src_path, "components", "comp1.yaml"))
    load_component_mock.assert_any_call(source=os.path.join(src_path, "components", "subdir", "comp2.yaml"))
    ml_client_mock.components.create_or_update.assert_any_call(comp_asset_mock)

@patch("src.scripts.register_ml_service_assets.os.walk")
@patch("src.scripts.register_ml_service_assets.os.path.isfile")
@patch("src.scripts.register_ml_service_assets.load_component")
def test_register_components_ignore(load_component_mock, isfile_mock, walk_mock):
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
    ml_client_mock.components.create_or_update.assert_not_called()


