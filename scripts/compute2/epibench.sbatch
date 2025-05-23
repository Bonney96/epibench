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

echo "Processing data for AML_263578..."
epibench process-data --config config/train_config_aml_263578.yaml -o processed_data/AML_263578_region_based

echo "Starting EpiBench training for AML_263578..."
epibench train --config config/train_config_aml_263578.yaml --output-dir output/training_run_AML_263578

echo "Making predictions for AML_263578..."
epibench predict --config config/train_config_aml_263578.yaml --checkpoint /storage2/fs1/dspencer/Active/spencerlab/abonney/epibench/checkpoints/epibench_experiment_20250516-235642/best_model.pth --input-data processed_data/AML_263578_region_based/test.h5 -o output/predictions/AML_263578

echo "Evaluating model for AML_263578..."
epibench evaluate --config config/train_config_aml_263578.yaml --checkpoint /storage2/fs1/dspencer/Active/spencerlab/abonney/epibench/checkpoints/epibench_experiment_20250516-235642/best_model.pth --test-data processed_data/AML_263578_region_based/test.h5 -o output/evaluation_results/AML_263578

echo "Interpreting model for AML_263578..."
epibench interpret --config config/train_config_aml_263578.yaml --checkpoint /storage2/fs1/dspencer/Active/spencerlab/abonney/epibench/checkpoints/epibench_experiment_20250516-235642/best_model.pth --input-data processed_data/AML_263578_region_based/test.h5 -o output/interpretation_results/AML_263578

# run sbatch script
# sbatch epibench.sbatch
