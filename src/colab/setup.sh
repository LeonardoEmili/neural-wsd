#!/bin/bash

# Downloads miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
export PATH="/root/miniconda3/bin:${PATH}"
conda init

# Creates the environment
echo "Creating the environment"
source ~/miniconda3/etc/profile.d/conda.sh
conda create -qyn neural-wsd python=3.9.7
conda activate neural-wsd
pip install -r /content/neural-wsd/requirements.txt

# Configure vscode and overwrite default settings
code --install-extension ms-python.python
cp /content/neural-wsd/src/colab/settings.json /root/.vscode-server/data/Machine/settings.json