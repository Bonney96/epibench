#!/bin/bash
#SBATCH --job-name=comp_interp
#SBATCH --partition=spencer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=comp_interp_%j.out
#SBATCH --error=comp_interp_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andrew.bonney@ucdenver.edu

# --- Environment Setup ---
echo "Setting up the environment..."
source /curc/sw/anaconda3/latest
conda activate /projects/abonney@xsede.org/conda/epibench

# --- Configuration ---
# This section defines all the paths and parameters for the workflow.
# TODO: This script is currently configured for a single sample.
#       Future work could involve looping over a list of sample IDs.

# The sample ID to process
SAMPLE_ID="263578" # Example: "263578"

# Base directory for pipeline outputs
PIPELINE_OUTPUT_BASE="/storage2/fs1/dspencer/Active/spencerlab/abonney/epibench/pipeline_output"

# Directory containing the split h5 files (train, val, test)
# This is typically the 'processed_data' directory for the feature set.
INPUT_DATA_DIR="${PIPELINE_OUTPUT_BASE}/processed_data/cpgislands"

# Directory to store intermediate and final results of this workflow
# This will contain the combined h5 file, predictions, and interpretation plots.
WORKFLOW_OUTPUT_DIR="${PIPELINE_OUTPUT_BASE}/comparative_interpretation/${SAMPLE_ID}"

# Paths to the trained model checkpoints
AML_MODEL_CHECKPOINT="/storage2/fs1/dspencer/Active/spencerlab/abonney/epibench/checkpoints/epibench_experiment_20250605-103939/best_model.pth"
CD34_MODEL_CHECKPOINT="/storage2/fs1/dspencer/Active/spencerlab/abonney/epibench/checkpoints/epibench_experiment_20250530-100744/best_model.pth"

# Path to the combined HDF5 file that will be created
COMBINED_H5_PATH="${WORKFLOW_OUTPUT_DIR}/${SAMPLE_ID}_all_regions.h5"

# Paths for the prediction outputs
AML_PREDICTIONS_PATH="${WORKFLOW_OUTPUT_DIR}/${SAMPLE_ID}_aml_predictions.h5"
CD34_PREDICTIONS_PATH="${WORKFLOW_OUTPUT_DIR}/${SAMPLE_ID}_cd34_predictions.h5"

# --- Workflow Execution ---
echo "Starting Comparative Interpretation Workflow for Sample ID: ${SAMPLE_ID}"
mkdir -p ${WORKFLOW_OUTPUT_DIR}

# --- Phase 1: Data Preparation ---
echo "[Phase 1] Combining train, validation, and test HDF5 files..."
python scripts/combine_h5_files.py \
    --sample-id "${SAMPLE_ID}" \
    --input-dir "${INPUT_DATA_DIR}" \
    --output-dir "${WORKFLOW_OUTPUT_DIR}"

# Check if data combination was successful
if [ ! -f "${COMBINED_H5_PATH}" ]; then
    echo "Error: Combined HDF5 file was not created. Aborting."
    exit 1
fi
echo "Combined data saved to ${COMBINED_H5_PATH}"


# --- Phase 2: Prediction Generation ---
echo "[Phase 2] Generating predictions..."

# Generate predictions using the AML-trained model
echo "  - Running prediction with AML model..."
epibench predict \
    --config config/predict_config_example.yaml \
    --checkpoint "${AML_MODEL_CHECKPOINT}" \
    --input-data "${COMBINED_H5_PATH}" \
    --output-file "${AML_PREDICTIONS_PATH}"

# Generate predictions using the CD34-trained model
echo "  - Running prediction with CD34 model..."
epibench predict \
    --config config/predict_config_example.yaml \
    --checkpoint "${CD34_MODEL_CHECKPOINT}" \
    --input-data "${COMBINED_H5_PATH}" \
    --output-file "${CD34_PREDICTIONS_PATH}"

# Check if prediction files were created
if [ ! -f "${AML_PREDICTIONS_PATH}" ] || [ ! -f "${CD34_PREDICTIONS_PATH}" ]; then
    echo "Error: One or both prediction files were not created. Aborting."
    exit 1
fi
echo "Prediction files generated successfully."


# --- Phase 3: Enhanced Interpretation ---
echo "[Phase 3] Running enhanced interpretation..."

# Define the directory for interpretation results
INTERPRET_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR}/interpretation_results"
mkdir -p ${INTERPRET_OUTPUT_DIR}

# Run the enhanced interpret command
epibench interpret \
    --config config/interpret_config.example.yaml \
    --checkpoint "${AML_MODEL_CHECKPOINT}" \
    --input-data "${COMBINED_H5_PATH}" \
    --secondary-predictions "${CD34_PREDICTIONS_PATH}" \
    --output-dir "${INTERPRET_OUTPUT_DIR}"

echo "Workflow completed. Results are in ${WORKFLOW_OUTPUT_DIR}" 