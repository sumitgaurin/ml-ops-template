#!/bin/bash
python --version

# Create a virtual environment
python -m venv venv
source venv/bin/activate
# for Windows: venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
python -m pip --version

# Upgrade Azure CLI and Azure ML SDK
pip install --upgrade azure-cli
pip install --upgrade azureml-sdk

# Install requirements
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
FILE="$SCRIPT_DIR/requirements.txt"

if [ -f "$FILE" ]; then
    echo "requirement.txt found"
    pip install -r "$FILE"
else
    echo "No python requirement definition found"
fi
