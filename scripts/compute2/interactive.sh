# docs
# https://washu.atlassian.net/wiki/spaces/RUD/pages/1720615105/Compute2+Quickstart

# interactive.sh - Interactive job launcher for WashU RIS Compute2
#
# This script provides example srun commands for launching interactive GPU jobs
# on the Compute2 cluster, with or without containers. Edit as needed for your workflow.

# launch an interactive job (with image)
# srun --gpus=4 --mem=64G --container-image="dhspence/quarto:v5" --container-mounts="/storage2/fs1/dspencer/Active:/storage2/fs1/dspencer/Active,/storage1/fs1/dspencer/Active:/storage1/fs1/dspencer/Active,/scratch2/fs1/dspencer:/scratch2/fs1/dspencer" --container-workdir=$PWD --pty /bin/bash
# srun --gpus=4 \
#      --mem=64G \
#      --exclude=c2-gpu-003,c2-gpu-004,c2-gpu-011 \
#      --container-image="dhspence/quarto:v5" \
#      --container-mounts="/storage2/fs1/dspencer/Active:/storage2/fs1/dspencer/Active,/storage1/fs1/dspencer/Active:/storage1/fs1/dspencer/Active,/scratch2/fs1/dspencer:/scratch2/fs1/dspencer" \
#      --container-workdir="$PWD" \
#      --pty /bin/bash

# launch an interactive job (without image)
srun --gpus=4 --mem=64G --exclude=c2-gpu-003,c2-gpu-004,c2-gpu-011,c2-gpu-009 --pty /bin/bash


# load required modules
# Export the custom module path
export MODULEPATH=/storage2/fs1/dspencer/Active/spencerlab/apps/modules/modulefiles:$MODULEPATH

# load basetools module
module load labtools



# Clone the repository:
git clone https://github.com/Bonney96/epibench.git
cd epibench

# Set up a Python virtual environment (Recommended):
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies:
pip install -r requirements.txt

# Install EpiBench in development mode:
pip install -e .

# Verify the installation:
epibench --version
epibench --help



# jypter notebook
jupyter lab --allow-root --no-browser --ip='*' --NotebookApp.token='' --NotebookApp.password=''
# get ip of the node
# [mmohamed@c2-bigmem-001 tools_testing]$ hostname -I
# 172.16.0.42 10.25.20.42 (take 2nd ip address)
# jupyter addres
10.25.20.31:8888

