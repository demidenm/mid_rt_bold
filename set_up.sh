#!/bin/bash
 
# Check if uv is installed
if ! command -v uv &> /dev/null; then
	echo "uv not found. Installing..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
else
	echo "uv is already installed."
fi


# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq not found. Installing..."
    if command -v conda &> /dev/null; then
        conda install -y -c conda-forge jq
    else
        echo "No supported package manager found for installing jq."
        exit 1
    fi
else
    echo "jq is already installed."
fi

if ! command -v fsl &> /dev/null && [ ! -d /usr/local/fsl ] && [ ! -d /opt/fsl ]; then
    echo "WARNING: FSL not found in the system. FSL is required for neuroimaging analysis. Randomise cannot be ran."
    echo "Please install FSL from https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation"
fi

# Proceed with setting up the environment based on pyproject.toml
source $HOME/.local/bin/env
uv sync
