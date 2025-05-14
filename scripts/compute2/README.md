# WashU Compute2 Scripts

This directory contains scripts specifically for use on the WashU RIS Compute2 cluster. These scripts are tailored for job submission, interactive sessions, and Jupyter notebook launching on the Compute2 environment.

## Included Scripts

- `epibench.sbatch`: Slurm batch script for running EpiBench training jobs.
- `epibench_notebook.sbatch`: Slurm batch script for launching a Jupyter Lab server on a compute node.
- `interactive.sh`: Script for launching an interactive shell session on a compute node, with or without a container.

## Usage

- Edit the scripts as needed for your project paths and resource requirements.
- Submit batch jobs with `sbatch <script>.sbatch`.
- Start interactive sessions with `bash interactive.sh` or by copying the `srun` command inside.

See comments in each script for more details and usage examples. 