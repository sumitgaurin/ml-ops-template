#!/bin/bash

# Confirm the environment is active
echo "Current python version"
python --version

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
python -m pip --version

# Upgrade Azure CLI and Azure ML SDK
echo "Upgrading azure CLI..."
pip install --upgrade azure-cli
echo "Upgrading azureml SDK..."
pip install --upgrade azureml-sdk
echo "Upgrading azureml sdk v2..."
pip install --upgrade azure-ai-ml

# Install requirements
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
FILE="$SCRIPT_DIR/requirements.txt"

if [ -f "$FILE" ]; then
    echo "requirement.txt found at $FILE"
	echo "installing dependencies..."
    pip install -r "$FILE"
else
    echo "No python requirement definition found"
fi