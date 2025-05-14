#!/bin/bash
#
# epibench_notebook.sbatch - Slurm batch script for Jupyter Lab on WashU RIS Compute2
#
# This script launches a Jupyter Lab server on a Compute2 GPU node for interactive work.
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

cd //storage2/fs1/dspencer/Active/spencerlab/abonney/epibench/jupyter_test_home

# Extract second IP and print Jupyter Lab URL
ip=$(hostname -I | awk '{print $2}')
if [[ -n "$ip" ]]; then
    echo "Jupyter Lab starting at: http://$ip:8888"
else
    echo "Could not determine IP address."
fi
# echo the working directory
echo "Working directory: $(pwd)"

jupyter lab --allow-root --no-browser --ip='*' --NotebookApp.token='' --NotebookApp.password=''

# run sbatch script
# sbatch epibench_notebook.sbatch
