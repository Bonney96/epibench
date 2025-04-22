# -*- coding: utf-8 -*-
"""CLI command for evaluating trained EpiBench models."""

import argparse
import logging
import sys
import os
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm # For progress bar

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from epibench.config.config_manager import ConfigManager
from epibench.utils.logging import LoggerManager
from epibench.models import models
from epibench.data.data_loader import create_dataloaders # Ensure this can create a test loader
from epibench.training.trainer import Trainer # For static load_model method
from epibench.evaluation import (
    calculate_regression_metrics, 
    plot_predictions_vs_actual, 
    plot_residuals
)

logger = logging.getLogger(__name__)

def setup_evaluate_parser(parser: argparse.ArgumentParser):
    """Adds arguments specific to the evaluate command."""
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the YAML/JSON configuration file used during training (or a similar one)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file (.pth) to evaluate."
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=False, # Make optional if test data path is in config
        help="Path to the test dataset file (e.g., HDF5). Overrides config if provided."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results (metrics.json, plots)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        help="Batch size for evaluation. Overrides config if provided."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Auto-detect by default
        help="Device to use ('cpu' or 'cuda'). Defaults to cuda if available."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Disable generation of performance plots."
    )

def evaluate_main(args):
    """Main function for the evaluate command."""
    
    # --- Helper function for JSON serialization ---
    def convert_numpy_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                            np.float64)):
            # Convert NaN to None for JSON compatibility
            return None if np.isnan(obj) else float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)): # Handle structured array scalars if they appear
            return None 
        elif isinstance(obj, dict):
            # Recursively process dictionary values
            return {k: convert_numpy_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process list items
            return [convert_numpy_to_json_serializable(item) for item in obj]
        else:
            # Return unchanged if not a numpy type or container we handle
            return obj

    # --- Load Config (Subtask 10.2 - Already partially done) ---
    config_manager = None
    config = {}
    try:
        config_manager = ConfigManager(args.config)
        config = config_manager.config
    except FileNotFoundError:
        logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
        logger.error(f"Error loading configuration from {args.config}: {e}", exc_info=True)
        sys.exit(1)

    # --- Setup Logging (using config) ---
    LoggerManager.setup_logger(config_manager=config_manager)
    logger.info("Starting EpiBench evaluation...")
    logger.info(f"Evaluation arguments: {args}")
    logger.info(f"Loaded configuration from: {args.config}")
    logger.debug(f"Full configuration: {config}")

    # --- Setup Output Directory ---
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {args.output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {args.output_dir}: {e}", exc_info=True)
        sys.exit(1)

    # --- Setup Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Model from Checkpoint (Subtask 10.3) ---
    model = None
    try:
        model_name = config.get('model', {}).get('name')
        model_params = config.get('model', {}).get('params', {})
        if not model_name:
            raise ValueError("Model name ('model.name') not found in configuration.")
        
        logger.info(f"Instantiating model architecture: {model_name}")
        ModelClass = models.get_model(model_name)
        model = ModelClass(**model_params) # Instantiate architecture

        logger.info(f"Loading model weights from checkpoint: {args.checkpoint}")
        # Use the static method from Trainer to load state
        model, _, _, checkpoint_info = Trainer.load_model(
            checkpoint_path=args.checkpoint, 
            model=model, 
            device=device
        ) 
        logger.info(f"Model loaded successfully from epoch {checkpoint_info.get('epoch', 'N/A')}.")
        model.eval() # Set model to evaluation mode

    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing key in checkpoint or config during model loading: {e}", exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error during model loading: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Test Data (Subtask 10.3) ---
    test_loader = None
    try:
        # Determine test data path (CLI arg overrides config)
        test_data_path = args.test_data or config.get('data', {}).get('test_path')
        if not test_data_path:
            raise ValueError("Test data path ('data.test_path') not specified in arguments or configuration.")
        
        # Determine batch size (CLI arg overrides config)
        batch_size = args.batch_size or config.get('data', {}).get('batch_size', 32)

        # Ensure the 'data' section exists in the config
        if 'data' not in config:
            config['data'] = {}

        # Update the main config dictionary with overrides for the data loader
        config['data']['batch_size'] = batch_size
        config['data']['test_path'] = test_data_path # Ensure the correct path is used
        config['data']['shuffle_test'] = False # Typically don't shuffle test data for evaluation

        logger.info(f"Loading test data from: {test_data_path} with batch size: {batch_size}")
        # Pass the full config dictionary to create_dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(config) # Unpack all three

        # Optional: Log if train/val loaders are unexpectedly created/returned
        if train_loader or val_loader:
            logger.debug("create_dataloaders returned train/validation loaders during evaluation, they will be ignored.")

        if test_loader is None:
             raise RuntimeError("create_dataloaders did not return a test loader.")

        logger.info("Test data loaded successfully.")

    except FileNotFoundError:
        logger.error(f"Test data file not found: {test_data_path}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error during data loading: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading test data: {e}", exc_info=True)
        sys.exit(1)

    # --- Evaluation Loop (Subtask 10.4) ---
    logger.info("Running evaluation loop...")
    all_y_true = []
    all_y_pred = []
    try:
        with torch.no_grad(): # Disable gradient calculations
            for batch in tqdm(test_loader, desc="Evaluating"): 
                # Assuming batch is a tuple/list: (inputs, targets)
                # Adjust based on your actual DataLoader output structure
                try:
                    inputs, targets = batch 
                except ValueError:
                     logger.error("Unexpected batch structure from DataLoader. Expected (inputs, targets). Skipping batch.")
                     continue
                
                inputs = inputs.to(device)
                # targets don't necessarily need to go to device if only used for metrics on CPU

                outputs = model(inputs)
                
                # Append results (move to CPU if needed)
                all_y_true.append(targets.cpu().numpy())
                all_y_pred.append(outputs.cpu().numpy())
                
        # Concatenate results from all batches
        if not all_y_true:
             logger.error("No results collected during evaluation loop. Check data loader and batch structure.")
             sys.exit(1)
             
        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)
        logger.info(f"Evaluation loop completed. Processed {len(all_y_true)} samples.")

    except Exception as e:
        logger.error(f"Error during evaluation loop: {e}", exc_info=True)
        sys.exit(1)

    # --- Calculate Metrics (Subtask 10.4) ---
    logger.info("Calculating metrics...")
    metrics = calculate_regression_metrics(all_y_true, all_y_pred)
    serializable_metrics = None # Initialize
    if metrics:
        # Convert numpy types for JSON serialization BEFORE logging/saving
        try:
            # Use a copy in case the original metrics dict is needed elsewhere with numpy types
            serializable_metrics = convert_numpy_to_json_serializable(metrics.copy()) 
            
            # Format metrics for logging nicely using the converted dict
            metrics_log_str = json.dumps(serializable_metrics, indent=2)
            logger.info(f"Calculated Metrics:\n{metrics_log_str}")
        except Exception as e:
             logger.error(f"Failed to convert metrics for JSON serialization: {e}", exc_info=True)
             serializable_metrics = None # Ensure it's None if conversion fails
             logger.error("Skipping results saving and plotting due to conversion error.")
             
    else:
        logger.error("Failed to calculate metrics. Skipping results saving and plotting.")
        # Optionally exit, or just proceed without results
        # sys.exit(1) 

    # --- Save Results (Subtask 10.5) ---
    # Use the serializable_metrics if conversion was successful
    if serializable_metrics: 
        metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
        try:
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=4) # Save the converted dict
            logger.info(f"Metrics saved to: {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics to {metrics_path}: {e}", exc_info=True)

    # --- Generate Plots (Subtask 10.5) ---
    # Check serializable_metrics existence for consistency, as plotting uses original arrays anyway
    if not args.no_plots and serializable_metrics is not None: 
        logger.info("Generating plots...")
        plot_pred_path = os.path.join(args.output_dir, "predictions_vs_actual.png")
        plot_resid_path = os.path.join(args.output_dir, "residuals.png")
        
        # Plotting functions should still use the original numpy arrays
        try:
            plot_predictions_vs_actual(all_y_true, all_y_pred, save_path=plot_pred_path)
        except Exception as e:
            logger.error(f"Failed to generate/save predictions vs actual plot: {e}", exc_info=True)
            
        try:
            plot_residuals(all_y_true, all_y_pred, save_path=plot_resid_path)
        except Exception as e:
            logger.error(f"Failed to generate/save residuals plot: {e}", exc_info=True)
            
    elif args.no_plots:
        logger.info("Plot generation disabled by --no-plots flag.")
    elif serializable_metrics is None: # Check if metrics calculation or conversion failed
        logger.warning("Skipping plot generation because metric calculation or conversion failed.")

    logger.info("EpiBench evaluation finished.")

if __name__ == '__main__':
    # This is for structuring the command if epibench becomes a multi-command tool
    # For now, we can assume this file is the entry point for 'evaluate'
    # In a real multi-command setup (e.g., using main.py), this would be different.
    parser = argparse.ArgumentParser(description="Evaluate an EpiBench model.")
    setup_evaluate_parser(parser)
    args = parser.parse_args()
    evaluate_main(args) 