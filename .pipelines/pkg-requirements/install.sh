#!/bin/bash

# Function to activate a conda environment
create_and_activate_conda_env() {
	source "$(conda info --base)/etc/profile.d/conda.sh"
	echo "Checking if conda environment $env_name exists"
	if ! conda info --envs | grep -q $env_name; then
		echo "Checking if environment azureml_py310_sdkv2 exists"
		if conda info --envs | grep -q 'azureml_py310_sdkv2'; then
			echo "Clone environment azureml_py310_sdkv2 to $env_name"
			conda create --name $env_name --clone azureml_py310_sdkv2
		else
			echo "Creating new conda environment '$env_name' with python 3.10"
			conda create -n $env_name python=3.10 -y
		fi
    fi

	echo "Activating the conda environment '$env_name'..."
	conda activate $env_name
}

# Function to create and activate a virtual environment using venv
create_and_activate_venv() {    
    # Check if the virtual environment already exists
	# by checking the existance of the directory
    if [ ! -d "$env_name" ]; then
        echo "Creating virtual environment '$env_name' using virtualenv..." 
        python -m venv $env_name
    fi
    
    # Activate the virtual environment
    echo "Activating the '$env_name' virtual environment..."
    source $env_name/bin/activate
}

activate_python_environment() {
	# Check if conda is installed
	if command -v conda &> /dev/null; then
		echo "Anaconda is installed on this machine."
		create_and_activate_conda_env
	else
		echo "Anaconda is not installed on this machine."
		create_and_activate_venv
	fi
}


# first positional argument passed to the script
# if positional argument is unset or empty, use venv as the default value
env_name="${1:-venv}"

# Confirm the environment is active
echo "Current python version"
python --version

# activate the python environment
activate_python_environment

# Confirm the environment is active
echo "$env_name environment python version"
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
