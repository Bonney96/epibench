# EpiBench Interpretation Tutorial

This tutorial guides you through using the `epibench interpret` command to understand the predictions of your trained models using feature attribution methods, specifically Integrated Gradients.

## Overview

The `interpret` command calculates the importance or contribution of each input feature (e.g., DNA bases, histone marks at specific positions) towards the model's prediction for a given sample. This helps in identifying which parts of the input sequence or which epigenetic marks the model considers most influential.

Currently, the primary supported method is **Integrated Gradients**.

## Prerequisites

1.  **Trained Model:** You need a model checkpoint (`.pth` file) previously trained using `epibench train`.
2.  **Training Configuration:** The YAML configuration file used during the training of the model is required to reconstruct the model architecture.
3.  **Input Data:** An HDF5 file (`.h5`) containing the data samples you want to interpret. This data should have been processed using `epibench process-data` and must include genomic coordinates (`chrom`, `start`, `end`).
4.  **Interpretation Configuration:** A YAML configuration file specifying interpretation parameters.
5.  **(Optional) Ground Truth BigWig Files:** If you want to generate visualization plots including ground truth histone coverage, you need the paths to the relevant BigWig files.

## Interpretation Configuration (`interpret_config.yaml`)

You need to create a YAML configuration file to control the interpretation process. Here's an explanation based on the example (`config/interpret_config.example.yaml`):

```yaml
# --- Required --- #
# Path to the configuration file used during the *training* of the model.
# This is needed to reconstruct the model architecture.
training_config: path/to/training_output/train_config.yaml 

# --- Interpretation Settings --- #
interpretation:
  # Method to use.
  method: IntegratedGradients # Currently only 'IntegratedGradients' is supported

  # Parameters specific to Integrated Gradients.
  integrated_gradients:
    # Number of steps for the approximation integral (higher = more accurate but slower).
    n_steps: 50 # Default: 50
    
    # Baseline for comparison. Options:
    # - "zero": A tensor of zeros (common default).
    # - "random": Gaussian noise baseline.
    # - "custom": Load a baseline from the file specified below.
    baseline_type: "zero" # Default: "zero"
    
    # Path to custom baseline file (e.g., .npy or .h5).
    # Required and must be a valid file path ONLY if baseline_type is "custom".
    # custom_baseline_path: path/to/baseline_samples.npy
    
    # Index of the target output neuron to interpret.
    # For single-output regression models, this is typically 0.
    target_output_index: 0 # Default: 0

# --- Feature Extraction Settings (Optional) --- #
feature_extraction:
  # Use absolute attribution values for ranking/thresholding.
  use_absolute_value: true # Default: true
  
  # Option 1: Extract top K features (bases/positions) based on score.
  # Takes precedence over threshold if both are set.
  # top_k: 100 # Default: None
  
  # Option 2: Extract features with scores above this threshold.
  # Score is absolute if use_absolute_value is true.
  threshold: 0.1 # Default: None

# --- Output Settings (Optional) --- #
output:
  # Save the raw attribution scores (can be large).
  save_attributions: true # Default: true
  
  # Generate and save visualization plots.
  generate_plots: true # Default: true
  
  # Prefix for output filenames (e.g., attributions, plots).
  # Full path determined by CLI -o/--output-dir.
  filename_prefix: "interpretation" # Default: "interpretation"

# --- Visualization Settings (Required if output.generate_plots is true) --- #
visualization:
  # List of histone mark names for the ground truth plot.
  # Order determines plot order and MUST match paths list below.
  histone_names: 
    - H3K4me
    - H3K4me3
    # ... add other histone marks ...
  
  # List of paths to the ground truth BigWig files corresponding to histone_names.
  # REQUIRED if generate_plots is true.
  histone_bigwig_paths:
    - /path/to/H3K4me.bw
    - /path/to/H3K4me3.bw
    # ... add paths corresponding to names above ...

  # Optional: Resolution (dots per inch) for saved plot images.
  plot_dpi: 150 # Default: 150

  # Optional: Max number of individual sample plots to generate.
  # Set to 0 or null/None to disable individual plots.
  max_samples_to_plot: 20 # Default: 20

# --- Logging Settings (Optional) --- #
logging_config:
  level: INFO # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
  # file: path/to/interpret_log.log # Optional: Path to save log file
```

**Key Sections:**

*   **`training_config` (Required):** Path to the original configuration file used to train the model being interpreted.
*   **`interpretation` (Required):** Settings for the attribution method.
    *   `method`: Currently only `IntegratedGradients`.
    *   `integrated_gradients`: Parameters specific to IG (steps, baseline type, target index).
*   **`feature_extraction` (Optional):** Parameters to identify and save the most important features (positions) based on attribution scores (either top K or threshold).
*   **`output` (Optional):** Controls what gets saved.
    *   `save_attributions`: Save the raw HDF5 attribution data.
    *   `generate_plots`: Generate the multi-panel visualization plots.
    *   `filename_prefix`: Prefix used for all output files.
*   **`visualization` (Required if `generate_plots` is true):** Settings for generating plots.
    *   `histone_names`, `histone_bigwig_paths`: Define the ground truth histone tracks to plot and their corresponding BigWig file paths.
    *   `plot_dpi`, `max_samples_to_plot`: Control plot resolution and the number of sample plots generated.
*   **`logging_config` (Optional):** Standard logging configuration.

## Running Interpretation

Use the following command structure:

```bash
epibench interpret \
    --config /path/to/your/interpret_config.yaml \
    --checkpoint /path/to/your/model_checkpoint.pth \
    --input-data /path/to/your/interpret_data.h5 \
    --output-dir /path/to/save/results \
    [--batch-size <size>] \
    [--device <cpu|cuda:0|...>]
```

**Arguments:**

*   `--config` (`-c`): Path to your interpretation YAML config file (described above).
*   `--checkpoint`: Path to the trained model checkpoint (`.pth`).
*   `--input-data` (`-i`): Path to the HDF5 data file containing samples to interpret.
*   `--output-dir` (`-o`): Directory where all output files will be saved.
*   `--batch-size` (Optional): Override the batch size used during interpretation.
*   `--device` (Optional): Specify compute device (e.g., `cuda:0`, `cpu`). Defaults to CUDA if available.

## Output Files

Based on your configuration, the following files might be generated in the specified `--output-dir`:

1.  **`<prefix>_attributions.h5`** (if `output.save_attributions: true`)
    *   An HDF5 file containing the raw attribution scores and associated metadata.
    *   **Structure:**
        *   `/attributions`: Dataset (float32) of shape `(num_samples, seq_len, num_features)` containing attribution scores.
        *   `/coordinates/chrom`: Dataset (string) storing chromosome names.
        *   `/coordinates/start`: Dataset (int64) storing start coordinates.
        *   `/coordinates/end`: Dataset (int64) storing end coordinates.
        *   `/metadata`: Group containing attributes like method, parameters, file paths, epibench version, creation date, and the full interpretation config as a JSON string.

2.  **`<prefix>_extracted_features.tsv`** (if `feature_extraction.top_k` or `feature_extraction.threshold` is set)
    *   A tab-separated file listing the positions/features identified as most important.
    *   **Columns:** `sample_index`, `chrom`, `genomic_position`, `score` (attribution score), `rank` (if using top_k), `feature_index_in_window`, `window_start`, `window_end`.

3.  **`<prefix>_viz_sample_<i>.png`** (if `output.generate_plots: true`)
    *   Individual multi-panel plot images for each sample (up to `visualization.max_samples_to_plot`).
    *   Each plot shows:
        *   Heatmap of attribution scores for DNA and input histone features.
        *   Heatmap of ground truth histone coverage fetched from BigWig files.
        *   A track indicating the genomic region.

## Next Steps

*   Analyze the generated attribution scores and extracted features.
*   Use the visualization plots to gain insights into model behavior for specific genomic regions.
*   Compare interpretations across different models or datasets. 