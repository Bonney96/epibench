#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestration script to run a standard EpiBench workflow:
process-data -> train -> evaluate -> predict

Fixes applied (Task 34):
- Corrected TypeError in path construction (line ~74) by casting sample_name
  and other config-derived values to str() before joining with pathlib.Path.
- Corrected subprocess argument errors by replacing shorthand '-o' with
  '--output-dir' in epibench command calls (process-data, evaluate, predict).
"""

import argparse
import logging
import subprocess
import sys
import os
from pathlib import Path
import concurrent.futures
import yaml  # Add yaml import for reading sample config
import torch # Add torch import for GPU check
from epibench.validation.config_validator import validate_process_config, ProcessConfig # Import validator
import tempfile # Add tempfile import
import shutil # Add shutil import for cleanup
from datetime import datetime
from typing import Optional

# Import environment checker
from scripts.check_environment import main as run_env_check

# Basic logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: list[str]):
    """Runs a command using subprocess and logs the output."""
    logger.info(f"Running command: {' '.join(command)}")
    try:
        process = subprocess.run(
            command,
            check=True,  # Raise exception on non-zero exit code
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        logger.info(f"Command stdout:\n{process.stdout}")
        if process.stderr:
            logger.warning(f"Command stderr:\n{process.stderr}")
        logger.info("Command completed successfully.")
    except FileNotFoundError:
        logger.error(f"Error: The command '{command[0]}' was not found. Is epibench installed and in PATH?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Stdout:\n{e.stdout}")
        logger.error(f"Stderr:\n{e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

def create_temp_config(base_config_path: str, 
                       sample_name: str,
                       base_output_dir: Path, 
                       updates: dict) -> Optional[str]:
    """Loads a base YAML config, applies updates, saves to a temp file, returns temp file path."""
    temp_config_file_path = None
    try:
        with open(base_config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Apply updates from the updates dictionary
        # Example: updates = {'data.train_path': 'new/path', 'checkpoint_dir': 'other/path'}
        for key_path, value in updates.items():
            keys = key_path.split('.')
            d = config_data
            for key in keys[:-1]:
                d = d.setdefault(key, {}) # Create nested dicts if they don't exist
            d[keys[-1]] = str(value) # Ensure value is string

        # Create a temporary directory
        temp_dir = base_output_dir / str(sample_name) / "temp_configs"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Write updated config to a temporary file
        temp_config_path_obj = temp_dir / f"temp_{Path(base_config_path).name}"
        with open(temp_config_path_obj, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
            
        temp_config_file_path = str(temp_config_path_obj.resolve())
        logger.info(f"[{sample_name}] Created temporary config: {temp_config_file_path}")
        return temp_config_file_path

    except KeyError as e:
        logger.error(f"[{sample_name}] Error updating base config {base_config_path}: Missing expected key path {e} during update process.", exc_info=True)
        return None # Indicate failure
    except FileNotFoundError:
        logger.error(f"[{sample_name}] Base configuration file not found: {base_config_path}")
        return None # Indicate failure
    except Exception as e:
        logger.error(f"[{sample_name}] Error creating temporary config from {base_config_path}: {e}", exc_info=True)
        return None # Indicate failure

def run_pipeline_for_sample(sample_config: dict, base_output_dir: Path, overwrite: bool):
    """Runs the full process->train->eval->predict pipeline for a single sample configuration.
    
    Args:
        sample_config (dict): Configuration for the specific sample.
        base_output_dir (Path): Base directory for all outputs.
        overwrite (bool): If True, force reprocessing even if output files exist.
    """
    
    sample_name = sample_config.get('name')
    process_data_config = sample_config.get('process_data_config')
    train_config = sample_config.get('train_config')
    input_data_for_prediction = sample_config.get('input_data_for_prediction') # Optional

    # Optional overrides for subdirs and filenames from sample config
    processed_data_name = sample_config.get('processed_data_name', "processed_data")
    training_output_name = sample_config.get('training_output_name', "training_output")
    evaluation_output_name = sample_config.get('evaluation_output_name', "evaluation_output")
    prediction_output_name = sample_config.get('prediction_output_name', "prediction_output")
    test_data_filename = sample_config.get('test_data_filename', "test.h5")
    validation_data_filename = sample_config.get('validation_data_filename', "validation.h5") # Added for checking
    train_data_filename = sample_config.get('train_data_filename', "train.h5") # Added for checking
    checkpoint_filename = sample_config.get('checkpoint_filename', "best_model.pth")

    if not all([sample_name, process_data_config, train_config]):
        logger.error(f"Skipping sample due to missing required fields (name, process_data_config, train_config): {sample_config}")
        return False # Indicate failure

    logger.info(f"===== Starting Pipeline for Sample: {sample_name} =====")

    # --- Setup Paths ---
    # Explicitly cast sample_name to string for robust path construction
    sample_output_dir = base_output_dir / str(sample_name) 
    
    # Cast other config-derived names to string for robustness
    process_out_dir = sample_output_dir / str(processed_data_name)
    train_out_dir = sample_output_dir / str(training_output_name)
    eval_out_dir = sample_output_dir / str(evaluation_output_name)
    predict_out_dir = sample_output_dir / str(prediction_output_name)

    # Expected intermediate file paths for checking existence and later use
    train_h5_path = process_out_dir / str(train_data_filename)
    val_h5_path = process_out_dir / str(validation_data_filename)
    test_h5_path = process_out_dir / str(test_data_filename)
    checkpoint_path = train_out_dir / str(checkpoint_filename)
    predict_input_path = Path(input_data_for_prediction) if input_data_for_prediction else test_h5_path

    # Create directories (safe to run even if skipping processing)
    for path in [process_out_dir, train_out_dir, eval_out_dir, predict_out_dir]:
        path.mkdir(parents=True, exist_ok=True)
        # Avoid excessive logging in parallel runs, log main creation in main()
        # logger.info(f"Ensured output directory exists: {path}")

    try:
        # --- Step 1: Process Data ---
        logger.info(f"[{sample_name}] --- Starting Step 1: Process Data ---")
        
        # Check if output files exist and if overwrite is False
        files_exist = train_h5_path.is_file() and val_h5_path.is_file() and test_h5_path.is_file()
        
        if not overwrite and files_exist:
            logger.info(f"[{sample_name}] Output HDF5 files already exist in {process_out_dir}. Skipping process-data step.")
        else:
            if overwrite and files_exist:
                 logger.info(f"[{sample_name}] Output HDF5 files exist, but --overwrite flag is set. Reprocessing...")
            # If files don't exist or overwrite is True, run processing
            process_cmd = [
                "epibench", "process-data",
                "--config", process_data_config,
                "--output-dir", str(process_out_dir) # Use full argument name
            ]
            run_command(process_cmd)

        # --- Step 2: Train Model ---
        logger.info(f"[{sample_name}] --- Starting Step 2: Train Model ---")
        train_updates = {
            'data.train_path': train_h5_path,
            'data.val_path': val_h5_path,
            'data.test_path': test_h5_path, # Ensure test path is updated if needed by trainer
            'checkpoint_dir': train_out_dir # Use the key expected by Trainer
        }
        temp_train_config_path = create_temp_config(train_config, sample_name, base_output_dir, train_updates)
        
        if not temp_train_config_path:
            raise RuntimeError(f"Failed to create temporary training config for {sample_name}")
            
        train_cmd = [
            "epibench", "train",
            "--config", temp_train_config_path, # Use the temporary config file
        ]
        run_command(train_cmd)
        
        # Check for checkpoint *after* training command
        if not checkpoint_path.is_file():
            # Check if it was saved in the *intended* directory, not the default one
            alt_checkpoint_dir_name = f"epibench_experiment_{datetime.now().strftime('%Y%m%d')}" # Approximate default name structure
            alt_checkpoint_path = Path("checkpoints") / alt_checkpoint_dir_name / "best_model.pth" # Example default path
            if alt_checkpoint_path.is_file():
                 logger.warning(f"Checkpoint found in default location {alt_checkpoint_path}, not expected {checkpoint_path}. Check Trainer config.")
                 # Optionally attempt to move/copy it?
                 # shutil.move(alt_checkpoint_path, checkpoint_path)
            else:
                raise RuntimeError(f"Training step finished, but checkpoint not found: {checkpoint_path}")
        logger.info(f"[{sample_name}] Found checkpoint file: {checkpoint_path}")

        # --- Step 3: Evaluate Model ---
        logger.info(f"[{sample_name}] --- Starting Step 3: Evaluate Model ---")
        if not test_h5_path.is_file():
            raise RuntimeError(f"Test data not found for evaluation: {test_h5_path}")
        logger.info(f"[{sample_name}] Found test data file for evaluation: {test_h5_path}")
        
        # Create temp config for evaluation (reusing train config as base)
        eval_updates = {
            # Update data paths needed by evaluate (might load train/val too)
             'data.train_path': train_h5_path,
             'data.val_path': val_h5_path,
             'data.test_path': test_h5_path,
             # Evaluate command takes output dir as CLI arg, no need to set in config?
             # Check epibench evaluate --help if needed
        }
        # Use original train_config as the base for evaluation config updates
        temp_eval_config_path = create_temp_config(train_config, sample_name, base_output_dir, eval_updates)
        
        if not temp_eval_config_path:
            raise RuntimeError(f"Failed to create temporary evaluation config for {sample_name}")

        evaluate_cmd = [
            "epibench", "evaluate",
            # Pass the *temporary* config with updated data paths
            "--config", temp_eval_config_path, 
            "--checkpoint", str(checkpoint_path),
            "--test-data", str(test_h5_path), # This CLI arg might override config, but update config too for safety
            "--output-dir", str(eval_out_dir) # Evaluate takes output dir via CLI
        ]
        run_command(evaluate_cmd)

        # --- Step 4: Generate Predictions ---
        logger.info(f"[{sample_name}] --- Starting Step 4: Generate Predictions ---")
        if not predict_input_path.is_file():
             raise RuntimeError(f"Input data for prediction not found: {predict_input_path}")
        logger.info(f"[{sample_name}] Using input data for prediction: {predict_input_path}")

        # Create temp config for prediction (reusing train config as base)
        predict_updates = {
             # Update data paths needed by predict (might just be input_data_path?)
             # Let's assume it might still load train/val/test for context/metadata
             'data.train_path': train_h5_path,
             'data.val_path': val_h5_path,
             'data.test_path': test_h5_path,
             # Update the specific input if the key exists, otherwise rely on CLI
             # 'data.predict_input_path': predict_input_path, # Adjust key if needed
        }
        # Use original train_config as the base for prediction config updates
        temp_predict_config_path = create_temp_config(train_config, sample_name, base_output_dir, predict_updates)
        
        if not temp_predict_config_path:
             raise RuntimeError(f"Failed to create temporary prediction config for {sample_name}")

        predict_cmd = [
            "epibench", "predict",
            # Pass the *temporary* config with updated data paths
            "--config", temp_predict_config_path, 
            "--checkpoint", str(checkpoint_path),
            # Pass input data via CLI (might override config)
            "--input-data", str(predict_input_path),
            "--output-dir", str(predict_out_dir) # Predict takes output dir via CLI
        ]
        run_command(predict_cmd)

        logger.info(f"===== Pipeline Completed Successfully for Sample: {sample_name} =====")
        return True # Indicate success
    
    except Exception as e:
        # Catch exceptions from run_command or RuntimeErrors
        logger.error(f"===== Pipeline FAILED for Sample: {sample_name} =====")
        logger.error(f"Error during pipeline execution: {e}", exc_info=True) # Log traceback
        return False # Indicate failure

def main():
    parser = argparse.ArgumentParser(
        description="Run standard EpiBench pipelines for one or more samples, potentially in parallel.",
        formatter_class=argparse.RawTextHelpFormatter # Keep formatting for help text
        )

    # Input Modes: Either single sample via args or multiple via YAML config
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--samples-config",
        help="Path to a YAML file listing multiple sample configurations to run.\n"
             "Format:\n"
             "- name: sample_a\n"
             "  process_data_config: path/to/process_a.yaml\n"
             "  train_config: path/to/train_a.yaml\n"
             "  # Optional fields like input_data_for_prediction, *_name, *_filename\n"
             "- name: sample_b\n"
             "  ...\n"
    )
    input_group.add_argument(
        "--single-sample-name",
        help="Run for a single sample specified by command-line arguments."
    )

    # Arguments required only if running for a single sample
    single_sample_group = parser.add_argument_group('Single Sample Arguments (if --single-sample-name is used)')
    single_sample_group.add_argument("--process-data-config", help="Path to configuration YAML/JSON for the 'process-data' step.")
    single_sample_group.add_argument("--train-config", help="Path to configuration YAML/JSON for the 'train' step.")
    single_sample_group.add_argument("--input-data-for-prediction", help="Optional: Path to specific input data for the final prediction step.")
    # Optional overrides for single sample
    single_sample_group.add_argument("--processed-data-name", default="processed_data", help="Subdirectory name for processed data.")
    single_sample_group.add_argument("--training-output-name", default="training_output", help="Subdirectory name for training outputs.")
    single_sample_group.add_argument("--evaluation-output-name", default="evaluation_output", help="Subdirectory name for evaluation results.")
    single_sample_group.add_argument("--prediction-output-name", default="prediction_output", help="Subdirectory name for prediction results.")
    single_sample_group.add_argument("--test-data-filename", default="test.h5", help="Expected filename of the test dataset.")
    single_sample_group.add_argument("--checkpoint-filename", default="best_model.pth", help="Expected filename of the best model checkpoint.")

    # General arguments
    parser.add_argument("--output-dir", required=True, help="Base directory for all pipeline outputs.")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of parallel processes to use when running multiple samples.")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation checks."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force reprocessing of data even if output HDF5 files already exist."
    )

    args = parser.parse_args()

    # --- Environment Validation ---
    if not args.skip_validation:
        logger.info("--- Running Environment Validation ---")
        try:
            run_env_check() # Exits with 0 on success, 1 on failure
            logger.info("Environment validation passed.")
        except SystemExit as e:
            if e.code == 0:
                # Expected exit on success
                logger.info("Environment validation passed (check script exited).")
            else:
                # Exit code 1 means validation failed
                logger.error("Environment validation failed. Exiting.")
                sys.exit(e.code) # Propagate the failure exit code
        except Exception as e:
            logger.error(f"An unexpected error occurred during environment validation: {e}", exc_info=True)
            sys.exit(1)
        logger.info("-------------------------------------")
    else:
        logger.warning("Skipping environment validation as requested.")

    # --- Determine Samples to Run ---
    samples_to_run = []
    if args.samples_config:
        logger.info(f"Loading sample configurations from: {args.samples_config}")
        try:
            with open(args.samples_config, 'r') as f:
                samples_to_run = yaml.safe_load(f)
            if not isinstance(samples_to_run, list):
                raise ValueError("Samples config file should contain a list of sample dictionaries.")
            logger.info(f"Found {len(samples_to_run)} samples in config file.")
        except FileNotFoundError:
            logger.error(f"Samples configuration file not found: {args.samples_config}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing samples YAML configuration file: {e}")
            sys.exit(1)
        except ValueError as e:
            logger.error(f"Invalid format in samples configuration file: {e}")
            sys.exit(1)

    elif args.single_sample_name:
        logger.info(f"Running for single sample: {args.single_sample_name}")
        # Validate that required single-sample args are provided
        if not args.process_data_config or not args.train_config:
             parser.error("--process-data-config and --train-config are required when using --single-sample-name.")
        
        single_sample_config = {
            'name': args.single_sample_name,
            'process_data_config': args.process_data_config,
            'train_config': args.train_config,
            'input_data_for_prediction': args.input_data_for_prediction,
            'processed_data_name': args.processed_data_name,
            'training_output_name': args.training_output_name,
            'evaluation_output_name': args.evaluation_output_name,
            'prediction_output_name': args.prediction_output_name,
            'test_data_filename': args.test_data_filename,
            'checkpoint_filename': args.checkpoint_filename,
        }
        samples_to_run.append(single_sample_config)

    if not samples_to_run:
        logger.error("No samples specified to run. Use --samples-config or --single-sample-name.")
        sys.exit(1)

    # --- Setup Base Output Dir ---
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using base output directory: {base_output_dir}")

    # --- Check GPU availability vs max_workers ---
    gpu_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if gpu_available else 0
    
    if gpu_available:
        logger.info(f"Detected {num_gpus} CUDA-enabled GPU(s).")
        if args.max_workers > num_gpus:
             logger.warning(f"Warning: max_workers ({args.max_workers}) is greater than the number of detected GPUs ({num_gpus}).")
             logger.warning("If pipeline steps are GPU-intensive, this may lead to resource contention or out-of-memory errors.")
             logger.warning("Consider setting --max-workers <= {num_gpus}.")
    else:
        logger.info("No CUDA-enabled GPU detected. Running on CPU.")
        
    # --- Execute Pipelines ---
    max_workers = args.max_workers
    if max_workers <= 0:
        max_workers = 1 
    
    logger.info(f"Starting pipeline execution for {len(samples_to_run)} samples using up to {max_workers} parallel processes.")

    successful_samples = 0
    failed_samples = 0

    # Use ProcessPoolExecutor for parallel execution
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        # Pass base_output_dir and args.overwrite to the worker function
        future_to_sample = {executor.submit(run_pipeline_for_sample, sample, base_output_dir, args.overwrite): sample for sample in samples_to_run}
        
        for future in concurrent.futures.as_completed(future_to_sample):
            sample_info = future_to_sample[future]
            sample_name = sample_info.get('name', 'unknown_sample')
            try:
                result = future.result()  # Get result (True for success, False for failure)
                if result:
                    successful_samples += 1
                    logger.info(f"Pipeline finished successfully for sample: {sample_name}")
                else:
                    failed_samples += 1
                    # Error details should have been logged within run_pipeline_for_sample
                    logger.warning(f"Pipeline finished with errors for sample: {sample_name}") 
            except Exception as exc:
                failed_samples += 1
                # Log exception if the future itself raised one (unexpected error in executor or task function)
                logger.error(f"Sample {sample_name} generated an exception during execution: {exc}", exc_info=True)

    logger.info("--- Pipeline Execution Summary ---")
    logger.info(f"Total Samples: {len(samples_to_run)}")
    logger.info(f"Successful: {successful_samples}")
    logger.info(f"Failed: {failed_samples}")
    logger.info("---------------------------------")

    if failed_samples > 0:
        logger.error("Some pipelines failed. Please review the logs above for details.")
        sys.exit(1)
    else:
        logger.info("All pipelines completed successfully.")

if __name__ == "__main__":
    main() 