# EpiBench Logging System Testing Guide

## Overview
This document provides testing instructions for the EpiBench logging system implemented in Task 38. The logging system tracks pipeline executions for reproducibility and analysis.

## Components to Test

### 1. Log Schema (epibench/logging/log_schema.py)
- Validates the JSON schema structure
- Tests empty log creation
- Tests timestamp formatting

### 2. LogManager (epibench/logging/log_manager.py)
- Tests log creation, updating, and finalization
- Tests atomic write operations
- Tests system information collection
- Tests thread safety with concurrent access

### 3. Pipeline Integration (epibench/pipeline/pipeline_executor.py)
- Tests logging during pipeline execution
- Tests error handling and partial logs
- Tests configuration capture

## Testing Steps

### Unit Tests
```python
# Test log schema validation
from epibench.logging import LogSchema
log = LogSchema.create_empty_log()
is_valid, error = LogSchema.validate_log(log)
assert is_valid

# Test LogManager
from epibench.logging import LogManager
log_mgr = LogManager("/tmp/test_logs")
exec_id = log_mgr.create_log("test_sample", "/tmp/output")
log_mgr.update_log({"custom_metadata": {"test": "value"}})
log_mgr.finalize_log("completed")
```

### Integration Tests
1. Run a simple pipeline with a single sample:
```bash
python epibench/pipeline/pipeline_executor.py \
    --output-dir /tmp/test_pipeline \
    --sample-list samples.txt
```

2. Check log creation in `/tmp/test_pipeline/logs/`

3. Verify log contents contain:
   - Execution metadata with timestamps
   - Hardware/software information
   - Pipeline stage tracking
   - Configuration file paths

### Example temp_config Content
The system should capture temp_configs like:
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

### Error Scenarios to Test
1. Pipeline failure - verify error information is logged
2. Interrupted execution - verify partial logs are saved
3. Missing configuration files - verify graceful handling
4. Concurrent sample processing - verify thread safety

### Performance Testing
1. Measure execution time with/without logging
2. Check log file sizes for efficiency
3. Test with multiple concurrent samples

### Validation Checklist
- [ ] Logs are created in the correct directory
- [ ] Each sample gets a unique execution ID
- [ ] Hardware info is correctly captured
- [ ] Software versions are recorded
- [ ] Git commit hash is captured (if in git repo)
- [ ] Pipeline stages have timing information
- [ ] Errors are properly logged with tracebacks
- [ ] Logs are valid JSON and match schema
- [ ] Atomic writes prevent corruption
- [ ] Thread safety with concurrent access

## Next Steps for Implementation
The following subtasks remain:
1. Configuration Parameter Aggregation (38.4)
2. ResultsCollector Integration (38.5)
3. Query/Analysis API (38.6)
4. CLI Commands (38.7)
5. Documentation (38.8)
6. Visualization Foundations (38.9)

## Notes
- The current implementation processes samples individually for better logging granularity
- LogManager uses buffering to minimize I/O impact
- Pipeline executor maintains backward compatibility with checkpoints
- The temp_configs aggregation (subtask 38.4) will need to parse YAML files like the example provided 