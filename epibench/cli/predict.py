# -*- coding: utf-8 -*-
"""CLI command for generating predictions using trained EpiBench models."""

import argparse
import logging
import sys
import os
import torch
from torch.utils.data import DataLoader # Might need a specific loader for prediction data
import numpy as np
from tqdm import tqdm
import pandas as pd # For saving predictions

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from epibench.config import ConfigManager
from epibench.utils.logging import LoggerManager
from epibench.models import models # Assuming get_model exists
from epibench.data.data_loader import create_dataloaders # Need a way to load prediction data
from epibench.training.trainer import Trainer # To load model state

logger = logging.getLogger(__name__)

def setup_predict_parser(parser: argparse.ArgumentParser):
    """Adds arguments specific to the predict command."""
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
        help="Path to the model checkpoint file (.pth) to use for prediction."
    )
    parser.add_argument(
        "-i", "--input-data",
        type=str,
        required=True,
        help="Path to the input dataset file (e.g., HDF5) for which to generate predictions."
    )
    parser.add_argument(
        "-o", "--output-file",
        type=str,
        required=True,
        help="Path to save the predictions (e.g., predictions.csv)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        help="Batch size for prediction. Overrides config if provided."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Auto-detect by default
        help="Device to use ('cpu' or 'cuda'). Defaults to cuda if available."
    )
    # Potentially add arguments for output format (csv, tsv, etc.) later

def predict_main(args):
    """Main function for the predict command."""
    # --- Load Config (Subtask 11.2 - Placeholder) ---
    config_manager = None
    config = {}
    try:
        config_manager = ConfigManager(args.config)
        config = config_manager.config
    except Exception as e:
        logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
        logger.error(f"Error loading configuration from {args.config}: {e}", exc_info=True)
        sys.exit(1)

    # --- Setup Logging (using config) ---
    LoggerManager.setup_logger(config_manager=config_manager)
    logger.info("Starting EpiBench prediction...")
    logger.info(f"Prediction arguments: {args}")
    logger.info(f"Loaded configuration from: {args.config}")
    logger.debug(f"Full configuration: {config}")

    # --- Setup Output Directory (if needed) ---
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Ensured output directory exists: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}", exc_info=True)
            sys.exit(1)

    # --- Setup Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Model from Checkpoint (Subtask 11.3) ---
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
        # We only need the model, not optimizer/scheduler state for prediction
        model, _, _, checkpoint_info = Trainer.load_model(
            checkpoint_path=args.checkpoint,
            model=model,
            device=device,
            load_optimizer_state=False, # Don't load optimizer/scheduler
            load_scheduler_state=False
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

    # --- Load Input Data (Subtask 11.3) ---
    predict_loader = None
    try:
        # Determine batch size (CLI arg overrides config)
        batch_size = args.batch_size or config.get('data', {}).get('batch_size', 32)

        # Create a config structure suitable for create_dataloaders
        # Assuming create_dataloaders can handle a 'predict_path' or similar
        # or that we modify it later. For now, we create a config dict.
        data_config = config.get('data', {}).copy() # Get data section or empty dict
        data_config['batch_size'] = batch_size
        # Use a specific key for prediction data path to avoid conflict with train/val/test
        data_config['predict_path'] = args.input_data 
        data_config['shuffle_predict'] = False # Don't shuffle prediction data

        logger.info(f"Loading prediction input data from: {args.input_data} with batch size: {batch_size}")

        # Use create_dataloaders, assuming it can handle a 'predict' split
        # This might require modification of create_dataloaders later
        # It should return a loader yielding only inputs, or inputs and dummy targets
        _, _, predict_loader = create_dataloaders(data_config, splits=['predict']) # Pass 'predict' split

        if predict_loader is None:
             raise RuntimeError("create_dataloaders did not return a prediction loader for the 'predict' split.")

        logger.info("Prediction input data loaded successfully.")

    except FileNotFoundError:
        logger.error(f"Prediction input data file not found: {args.input_data}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error during data loading: {e}", exc_info=True)
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Data loading runtime error: {e}", exc_info=True)
        logger.error("This might indicate that create_dataloaders needs adjustment for prediction mode.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading prediction input data: {e}", exc_info=True)
        sys.exit(1)

    # Subtask 11.4: Prediction Loop
    logger.info("Running prediction loop...")
    all_predictions = []
    with torch.no_grad():
        for batch in predict_loader:
            inputs = batch # Adjust based on loader output for prediction
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)

    # Subtask 11.4: Save Predictions
    logger.info(f"Saving predictions to: {args.output_file}")
    try:
        # Assuming predictions should be saved in a simple format, e.g., CSV
        # Might need sample IDs or coordinates from the input data to pair with predictions
        pred_df = pd.DataFrame({'predictions': all_predictions.flatten()})
        pred_df.to_csv(args.output_file, index=False)
        logger.info("Predictions saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save predictions to {args.output_file}: {e}", exc_info=True)
        sys.exit(1)

    logger.info("EpiBench prediction finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate predictions using an EpiBench model.")
    setup_predict_parser(parser)
    args = parser.parse_args()
    predict_main(args) 