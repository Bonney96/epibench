# Tutorial: Training and Evaluating a Model

This tutorial guides you through the basic workflow of processing input data, training the `SeqCNNRegressor` model, and evaluating its performance using EpiBench.

## Prerequisites

*   EpiBench installed (see main `README.md`).
*   Raw input data files (e.g., BED file defining regions, BigWig files for sequence tracks and histone marks).
*   Configuration files for data processing and training (examples provided in `config/`).

## Step 1: Process Raw Data

First, we need to convert the raw genomic data into the matrix format required by the model. This involves defining the input files and processing parameters in a configuration file (e.g., `config/process_config_example.yaml`).

```yaml
# Example: config/process_config_example.yaml
input:
  regions_bed: path/to/your/genomic_regions.bed
  sequence_fasta: path/to/reference_genome.fasta # Or specify BigWig below
  tracks_bigwig: # List of BigWig files for sequence context and histone marks (11 tracks)
    - path/to/track1.bw
    - path/to/track2.bw
    # ... (up to 11 tracks)
  target_bigwig: path/to/methylation_target.bw # BigWig with target values (e.g., methylation)

processing:
  window_size: 10000
  num_tracks: 11
  # Add other processing parameters like aggregation methods if needed

splitting:
  method: chromosome # Use chromosome-based splitting
  split_ratios: [0.7, 0.15, 0.15] # Train/Validation/Test split
  # Or specify chromosomes directly:
  # train_chroms: ["chr1", "chr3", ...]
  # val_chroms: ["chr2", "chr8"]
  # test_chroms: ["chrX"]

output:
  # Output filenames are generated based on split type
  # e.g., train.h5, validation.h5, test.h5
  format: hdf5 # Or parquet
```

Run the `process-data` command, providing your configuration file and specifying an output directory:

```bash
epibench process-data \
    --config config/process_config_example.yaml \
    -o output/my_experiment/processed_data
```

This will create `train.h5`, `validation.h5`, and `test.h5` (or `.parquet`) files in the specified output directory (`output/my_experiment/processed_data`).

## Step 2: Train the Model

Next, configure the model training process. Define the model architecture, data paths, and training hyperparameters in a configuration file (e.g., `config/train_config_example.yaml`).

```yaml
# Example: config/train_config_example.yaml
data:
  train_path: output/my_experiment/processed_data/train.h5 # Path from Step 1
  val_path: output/my_experiment/processed_data/validation.h5 # Path from Step 1
  batch_size: 64
  num_workers: 4

model:
  name: SeqCNNRegressor
  params:
    input_channels: 11 # Number of input tracks
    seq_length: 10000 # Must match processing window size
    cnn_filters: [64, 128, 256] # Example filter counts
    kernel_sizes: [9, 25] # Example kernel sizes for branches
    fc_units: [512, 256]
    dropout_rate: 0.3
    use_batch_norm: true

training:
  optimizer: AdamW
  optimizer_params:
    lr: 0.001
    weight_decay: 0.01
  loss_function: MSELoss
  epochs: 50
  early_stopping_patience: 5
  device: cuda # Or cpu

# Optional: Hyperparameter Optimization (HPO) settings
hpo:
  enabled: false # Set to true to enable Optuna HPO
  n_trials: 20
  # Define search space for hyperparameters here if enabled
  # e.g., lr: [0.0001, 0.01]
```

Run the `train` command, providing the training configuration and an output directory for checkpoints and logs:

```bash
epibench train \
    --config config/train_config_example.yaml \
    --output-dir output/my_experiment/training_output
```

This will train the model, save checkpoints (including `best_model.pth`), and log training progress.

## Step 3: Evaluate the Model

Finally, evaluate the performance of the trained model on the test set.

Run the `evaluate` command, referencing the training configuration (for model/data setup), the best checkpoint from training, the test data from Step 1, and an output directory for results:

```bash
epibench evaluate \
    --config config/train_config_example.yaml \
    --checkpoint output/my_experiment/training_output/best_model.pth \
    --test-data output/my_experiment/processed_data/test.h5 \
    -o output/my_experiment/evaluation_results
```

This command will:
1.  Load the model specified in `train_config_example.yaml`.
2.  Load the weights from `best_model.pth`.
3.  Load the test data from `test.h5`.
4.  Run predictions on the test data.
5.  Calculate regression metrics (MSE, MAE, R², Pearson, Spearman by default).
6.  Save the metrics results (e.g., `evaluation_metrics.json`) and potentially visualizations (e.g., scatter plots) to the specified output directory (`output/my_experiment/evaluation_results`).

Review the saved metrics to assess your model's performance. 