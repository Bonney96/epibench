# EpiBench Logging System Documentation

## Overview

The EpiBench logging system provides robust run tracking, reproducibility, and analysis capabilities. It captures detailed metadata for every pipeline execution, supports advanced querying and analysis, and integrates with the EpiBench CLI for user-friendly log management.

---

## Architecture

- **LogManager**: Handles log creation, updates, and finalization during pipeline runs.
- **LogSchema**: Defines the JSON schema for logs, ensuring all required metadata is captured.
- **LogQuery**: Enables flexible querying, filtering, and exporting of logs.
- **LogAnalyzer**: Provides advanced statistical analysis and insights.
- **CLI Integration**: The `epibench logs` command exposes all log management features to users.

![Logging System Architecture](images/logging_architecture.png) <!-- Add diagram if available -->

---

## Log Schema

Logs are stored as JSON files with the following main sections:
- **execution_metadata**: Run ID, timestamps, EpiBench version, status
- **input_configuration**: Sample ID, regions file, config file paths
- **runtime_information**: Hardware/software specs, compute environment
- **pipeline_information**: Command-line args, pipeline stages, checkpoints
- **output_information**: Output directories and files
- **performance_metrics**: MSE, R², MAE, and other metrics
- **configuration_parameters**: Aggregated config (including temp_configs)
- **error_information**: Error details if run failed
- **custom_metadata**: User-defined metadata

### Example Log Entry
```json
{
  "log_version": "1.0.0",
  "execution_metadata": {
    "execution_id": "a1b2c3d4",
    "timestamp_start": "2024-06-01T12:00:00Z",
    "timestamp_end": "2024-06-01T12:30:00Z",
    "epibench_version": "1.2.3",
    "epibench_commit_hash": "abc1234",
    "execution_status": "completed"
  },
  "input_configuration": {
    "sample_id": "sample_001",
    "regions_file_path": "data/regions.bed",
    "config_files": {
      "process_config": "config/process_config.yaml",
      "train_config": "config/train_config.yaml"
    }
  },
  "runtime_information": {
    "duration_seconds": 1800,
    "compute_environment": "local",
    "hardware_specs": {"cpu_info": {"model": "Intel i7", "cores": 8, "threads": 16}, "memory_gb": 32},
    "software_environment": {"os_info": {"system": "Darwin", "release": "24.5.0", "version": "..."}, "python_version": "3.10.12"}
  },
  "pipeline_information": {
    "command_line_args": ["epibench", "train", "--config", "..."],
    "pipeline_stages": [{"name": "train", "status": "completed", "start_time": "...", "end_time": "..."}],
    "checkpoint_data": null
  },
  "output_information": {
    "base_output_directory": "output/training_run_01",
    "output_paths": {"evaluation_output": "...", "prediction_output": "..."}
  },
  "performance_metrics": {"mse": 0.012, "r_squared": 0.98, "mae": 0.01},
  "configuration_parameters": {
    "effective_config": {"model": {"name": "SeqCNNRegressor", "params": {"input_channels": 11}}},
    "temp_configs_content": {"train_config.yaml": {"epochs": 50, "batch_size": 64}}
  },
  "error_information": null,
  "custom_metadata": {"notes": "Test run"}
}
```

---

## CLI Usage Guide

The EpiBench CLI provides powerful log management commands:

### List Logs
```bash
epibench logs list --log-dir logs/ --status completed --format table
```
- Shows available logs with filtering, sorting, and pagination.

### Show Log Details
```bash
epibench logs show a1b2c3d4 --log-dir logs/ --section all --format rich
```
- Displays detailed information for a specific log.

### Search Logs
```bash
epibench logs search --metric "r_squared>0.9" --config "model.name=SeqCNNRegressor" --format table
```
- Finds logs matching metric thresholds and config parameters.

### Compare Logs
```bash
epibench logs compare a1b2c3d4 e5f6g7h8 --focus metrics --format table
```
- Side-by-side comparison of multiple runs.

### Export Logs
```bash
epibench logs export --format csv --output logs_export.csv --fields execution_id mse r_squared
```
- Exports logs to CSV, JSON, or Excel for external analysis.

### Analyze Logs
```bash
epibench logs analyze --analysis-type summary --metric r_squared --plot
```
- Runs statistical analysis and generates plots.

---

## Log Analysis Workflows & Examples

### Example: Identify Best Runs
```bash
epibench logs search --metric "r_squared>0.95" --format table
```

### Example: Compare Configurations
```bash
epibench logs compare run1 run2 --focus config --format table
```

### Example: Export for External Analysis
```bash
epibench logs export --format csv --output best_runs.csv --fields execution_id mse r_squared
```

---

## Configuration Aggregation & temp_configs

- The logging system aggregates all configuration files used in a run, including temporary configs generated during execution.
- Example temp_configs YAML:
```yaml
checkpoint_dir: pipeline_output/263578/training_output
data:
  batch_size: 64
  num_workers: 0
  shuffle_train: true
  shuffle_val: false
  test_path: pipeline_output/263578/processed_data/test.h5
  train_path: pipeline_output/263578/processed_data/train.h5
  val_path: pipeline_output/263578/processed_data/validation.h5
model:
  name: SeqCNNRegressor
  params:
    activation: ReLU
    dropout_rate: 0.4
    fc_units: [1024, 512]
    input_channels: 11
    kernel_sizes: [3, 9, 25, 51]
    num_filters: 64
    use_batch_norm: true
training:
  device: cuda
  early_stopping_patience: 7
  epochs: 50
  gradient_clipping: 1.0
  loss_function: MSELoss
  optimizer: AdamW
  optimizer_params:
    lr: 0.0005
    weight_decay: 0.01
```
- These parameters are accessible via the CLI and analysis tools for reproducibility and troubleshooting.

---

## Extending the Logging System

- To add new fields, update the schema in `epibench/logging/log_schema.py` and ensure LogManager populates them.
- To add new CLI commands, extend `epibench/cli/logs.py` following the existing command structure.
- Maintain backward compatibility by using optional fields and versioning.

---

## Testing & Validation

- Validate logs using:
```python
from epibench.logging import LogSchema
log = LogSchema.create_empty_log()
is_valid, error = LogSchema.validate_log(log)
assert is_valid
```
- Test log creation, updates, and analysis using the provided unit and integration tests.
- Use the checklist in `TESTING_NOTES_LOGGING.md` to ensure correctness and robustness. 