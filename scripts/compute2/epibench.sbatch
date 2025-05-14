#!/bin/bash
#
# epibench.sbatch - Slurm batch script for EpiBench on WashU RIS Compute2
#
# This script submits a GPU job to the Compute2 cluster for EpiBench training.
# Edit resource requests and paths as needed for your project.

#SBATCH --job-name=epibench
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --output=epibench.out
#SBATCH --error=epibench.err
#SBATCH --exclude=c2-gpu-003,c2-gpu-004,c2-gpu-011,c2-gpu-009


# Export the custom module path
export MODULEPATH=/storage2/fs1/dspencer/Active/spencerlab/apps/modules/modulefiles:$MODULEPATH

# load basetools module
module load labtools

# Activate virtual environment and run epibench commands
source /storage2/fs1/dspencer/Active/spencerlab/abonney/epibench/.venv/bin/activate

echo "Python version: $(python --version)"
echo "EpiBench version:"
epibench --version

echo "EpiBench help:"
epibench --help

echo "Starting EpiBench training for AML_263578..."
epibench train --config config/train_config_aml_263578.yaml

# run sbatch script
# sbatch epibench.sbatch
