# -*- coding: utf-8 -*-
"""CLI command for interpreting trained EpiBench models using Integrated Gradients."""

import argparse
import logging
import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import yaml
from typing import Optional # Add Optional typing hint
import h5py

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from epibench.config import config_manager
from epibench.utils.logging import LoggerManager
from epibench.models import models
from epibench.data.datasets import HDF5Dataset
from torch.utils.data import DataLoader
from epibench.validation.config_validator import validate_interpret_config, InterpretConfig
from epibench.interpretation.attributors import calculate_integrated_gradients, generate_baseline
from epibench.interpretation.io import save_interpretation_results, extract_and_save_features, generate_and_save_plots # Added IO imports

logger = logging.getLogger(__name__)

# --- Custom Collate Function ---
def interpret_collate_fn(batch):
    """Custom collate function for interpretation DataLoader.

    Handles batches where coordinates might be None or dicts.
    """
    # batch is a list of tuples: [(features1, target1, coords1), (features2, target2, coords2), ...]
    features_list = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]
    coords_list = [item[2] for item in batch] # Collect coords (should be list of dicts or Nones)

    # Use default_collate for features and targets (assumes they are tensors)
    features_collated = torch.utils.data.default_collate(features_list)
    targets_collated = torch.utils.data.default_collate(targets_list)

    # Return the collated tensors and the raw list of coordinates
    return features_collated, targets_collated, coords_list
# --- End Custom Collate Function ---

def setup_interpret_parser(parser: argparse.ArgumentParser):
    """Adds arguments specific to the interpret command."""
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help=(
            "Path to the YAML interpretation configuration file. This file specifies "
            "the attribution method, parameters, output options, visualization settings, "
            "and path to the original training configuration."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint file (.pth) to interpret."
    )
    parser.add_argument(
        "-i", "--input-data",
        type=str,
        required=True,
        help=(
            "Path to the input dataset file (HDF5 format) containing the samples "
            "for which to generate interpretations. Must include coordinate information."
        )
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory to save all interpretation results (e.g., attributions.h5, plots, features.tsv)."
    )
    parser.add_argument(
        "--secondary-predictions",
        type=str,
        required=False,
        help="Path to an HDF5 file containing secondary model predictions for comparative analysis (e.g., from a baseline model)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=None, # Default is now taken from config, CLI overrides
        help="Batch size for processing data during interpretation. Overrides the batch_size in the config if provided."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Auto-detect by default
        help="Compute device to use ('cpu' or 'cuda[:<index>]'). Defaults to 'cuda' if available, otherwise 'cpu'."
    )

def interpret_main(args):
    """Main function for the interpret command."""
    # Basic logging setup for early errors, will be overridden by config if successful
    logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Initialize logger early

    # --- Load and Validate Interpretation Config (using new validator) ---
    config: Optional[InterpretConfig] = None
    try:
        config = validate_interpret_config(args.config)
    except Exception as e:
        # logger is now guaranteed to be defined
        logger.error(f"Error loading or validating configuration from {args.config}: {e}", exc_info=True)
        sys.exit(1)

    # --- Setup Logging (using config) ---
    LoggerManager.setup_logger(
        config_manager=None,
        default_log_level=getattr(logging, config.logging_config.level, logging.INFO),
        default_log_file=config.logging_config.file,
        force_reconfigure=True
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting EpiBench model interpretation...")
    logger.info(f"Interpretation arguments: {args}")
    logger.info(f"Loaded configuration from: {args.config}")
    logger.debug(f"Validated configuration: {config.model_dump_json(indent=2)}")

    # --- Setup Output Directory ---
    try:
        # Use Path object for consistency
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}", exc_info=True)
        sys.exit(1)
    
    # --- Generate Filename Prefix ---
    # Create a consistent prefix based on input file and config for output files
    input_filename = Path(args.input_data).stem
    config_filename = Path(args.config).stem
    filename_prefix = f"{input_filename}_{config_filename}"
    logger.info(f"Using filename prefix for output files: {filename_prefix}")

    # --- Setup Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Model Architecture (using TRAINING config) ---
    # We need the training config to know the model architecture params
    # Note: This assumes training config format is loadable; might need a dedicated trainer config validator
    model = None
    training_config_path = config.training_config
    try:
        logger.info(f"Loading training configuration for model architecture: {training_config_path}")
        if not Path(training_config_path).exists():
             raise FileNotFoundError(f"Training configuration file not found: {training_config_path}")
        with open(training_config_path, 'r') as f:
             # Warning: Assuming training config is simple dict loadable by pyyaml
             # A proper solution would use a validated TrainingConfig Pydantic model
             train_config_data = yaml.safe_load(f) 
             if not train_config_data or 'model' not in train_config_data:
                 raise ValueError("Training config missing or does not contain 'model' section.")
            
        model_config = train_config_data['model']
        model_name = model_config.get('name')
        model_params = model_config.get('params', {})
        if not model_name:
            raise ValueError("Model name ('model.name') not found in configuration.")
        
        logger.info(f"Instantiating model architecture: {model_name}")
        ModelClass = models.get_model(model_name)
        model = ModelClass(**model_params)
        model.to(device) # Move model to device BEFORE loading state dict

        # --- Load Model Checkpoint --- 
        logger.info(f"Loading model weights from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle different checkpoint formats (e.g., containing 'model_state_dict')
        if 'model_state_dict' in checkpoint:
             model_state_dict = checkpoint['model_state_dict']
             epoch = checkpoint.get('epoch', 'N/A')
             logger.info(f"Loading model_state_dict from epoch {epoch}.")
        elif 'state_dict' in checkpoint: # Another common pattern
            model_state_dict = checkpoint['state_dict']
            epoch = checkpoint.get('epoch', 'N/A')
            logger.info(f"Loading state_dict from epoch {epoch}.")
        else:
             # Assume the checkpoint *is* the state_dict
             model_state_dict = checkpoint
             logger.info("Loading entire checkpoint file as state_dict.")
            
        # Load the state dict
        model.load_state_dict(model_state_dict)
        logger.info(f"Model weights loaded successfully from {args.checkpoint}")
        model.eval() # IMPORTANT: Ensure model is in evaluation mode for interpretation

    except FileNotFoundError as e:
        logger.error(f"File not found during model/config loading: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing training config YAML file {training_config_path}: {e}", exc_info=True)
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

    # --- Load Input Data (using HDF5Dataset) ---
    interpret_loader = None
    dataset_len = 0
    try:
        interpret_data_path = Path(args.input_data) # Use Path object
        if not interpret_data_path.exists():
            raise FileNotFoundError(f"Interpretation input data file not found: {interpret_data_path}")
        
        # Note: Config object `config` is now InterpretConfig, doesn't have batch size directly.
        # Batch size is mainly a CLI concern for interpretation.
        batch_size = args.batch_size if args.batch_size is not None else config.interpretation.internal_batch_size # Use interpretation.internal_batch_size as default
        num_workers = 0 # Usually set to 0 for interpretation unless I/O is bottleneck

        logger.info(f"Loading interpretation input data from: {interpret_data_path} with batch size: {batch_size}")

        # Directly use HDF5Dataset and DataLoader
        interpret_dataset = HDF5Dataset(h5_path=interpret_data_path)
        dataset_len = len(interpret_dataset)
        if dataset_len == 0:
            raise ValueError(f"Input data file {interpret_data_path} contains 0 samples.")
            
        # Collate function might be needed if HDF5Dataset returns dicts/non-standard types
        # For now, assume default collate works if features/targets are tensors
        interpret_loader = DataLoader(
            interpret_dataset,
            batch_size=batch_size,
            shuffle=False, # Never shuffle interpretation data
            num_workers=num_workers, 
            pin_memory=True if device == torch.device('cuda') else False, # Explicit check for cuda device
            collate_fn=interpret_collate_fn # USE CUSTOM COLLATE FN
        )

        logger.info("Interpretation input data loaded successfully.")

    except FileNotFoundError as e:
        logger.error(f"File not found during data loading: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error during data loading: {e}", exc_info=True)
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Data loading runtime error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading interpretation input data: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Secondary Predictions (Optional) ---
    secondary_predictions = None
    if args.secondary_predictions:
        try:
            secondary_path = Path(args.secondary_predictions)
            if not secondary_path.exists():
                raise FileNotFoundError(f"Secondary predictions file not found: {secondary_path}")
            logger.info(f"Loading secondary predictions from: {secondary_path}")
            with h5py.File(secondary_path, 'r') as f:
                # Assuming the dataset is named 'predictions' as per the plan
                if 'predictions' not in f:
                    raise KeyError("Dataset 'predictions' not found in secondary predictions file.")
                secondary_predictions = f['predictions'][:]
            logger.info(f"Successfully loaded {len(secondary_predictions)} secondary predictions.")
            if len(secondary_predictions) != dataset_len:
                logger.warning(
                    f"Mismatch in sample count between primary data ({dataset_len}) and "
                    f"secondary predictions ({len(secondary_predictions)}). Ensure they correspond."
                )
        except Exception as e:
            logger.error(f"Failed to load secondary predictions: {e}", exc_info=True)
            sys.exit(1)

    # --- Run Interpretation Loop (Subtask 28.4) ---
    all_attributions = []
    all_coordinates = [] # Store coordinates corresponding to attributions
    all_predictions = []
    all_actuals = []
    interpret_params = config.interpretation
    ig_params = interpret_params.integrated_gradients

    logger.info("Calculating Integrated Gradients attributions...")
    try:
        for batch_idx, batch in enumerate(tqdm(interpret_loader, desc="Calculating Attributions", total=len(interpret_loader))):
             features, targets, coordinates_batch = batch # HDF5Dataset now returns coordinates
             features = features.to(device)
             targets = targets.to(device)

             # Get model predictions
             with torch.no_grad():
                 predictions = model(features)
             
             # Store predictions and actuals
             all_predictions.append(predictions.cpu().detach().numpy())
             all_actuals.append(targets.cpu().detach().numpy())

             # --- Add Debug Logging ---
             if batch_idx == 0: # Log only for the first batch to avoid spam
                 logger.debug(f"Batch {batch_idx} - features shape: {features.shape}")
                 if coordinates_batch:
                     logger.debug(f"Batch {batch_idx} - coordinates_batch type: {type(coordinates_batch)}")
                     logger.debug(f"Batch {batch_idx} - coordinates_batch length: {len(coordinates_batch)}")
                     # Check for None values within the batch list
                     none_count = sum(1 for c in coordinates_batch if c is None)
                     if none_count > 0:
                         logger.debug(f"Batch {batch_idx} - Found {none_count} None values in coordinates_batch!")

                     for i, coord_elem in enumerate(coordinates_batch):
                         # Only log details for the first few non-None elements
                         if i < 5 and coord_elem is not None:
                             logger.debug(f"Batch {batch_idx} - coord element {i} type: {type(coord_elem)}")
                             # Assuming coord_elem is a dict as per HDF5Dataset
                             if isinstance(coord_elem, dict):
                                  logger.debug(f"Batch {batch_idx} - coord element {i} value: {coord_elem}")
                             else:
                                 # Log if it's something unexpected
                                 logger.debug(f"Batch {batch_idx} - coord element {i} UNEXPECTED TYPE: {type(coord_elem)}, value: {coord_elem}")
                         elif coord_elem is None:
                             # Optionally log the index of None values if needed
                             # logger.debug(f"Batch {batch_idx} - coord element {i} is None")
                             pass # Avoid spamming for every None
                 else:
                     logger.debug(f"Batch {batch_idx} - No coordinates returned in batch (coordinates_batch is empty or None).")
             # --- End Debug Logging ---

             # --- Calculate Attributions for the Batch ---
             batch_attributions = calculate_integrated_gradients(
                 model=model,
                 inputs=features,
                 baseline=generate_baseline(
                     baseline_type=ig_params.baseline_type,
                     input_batch=features,
                     custom_path=ig_params.custom_baseline_path, 
                     seed=interpret_params.seed # Pass seed from interpretation params
                 ),
                 target_index=ig_params.target_output_index,
                 n_steps=ig_params.n_steps
             )
             
             # --- Collect Results --- 
             all_attributions.append(batch_attributions.cpu().detach().numpy())
             # Append coordinates for this batch - KEEP None values to maintain alignment
             all_coordinates.extend(coordinates_batch) # Extend with original batch, including None

    except Exception as e:
        logger.error(f"Error during attribution calculation: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Attribution calculation complete.")

    # --- Aggregate Results ---
    if not all_attributions:
        logger.error("No attributions were generated. Exiting.")
        sys.exit(1)
        
    try:
        logger.info("Aggregating attribution results...")
        final_attributions = np.concatenate(all_attributions, axis=0)
        final_predictions = np.concatenate(all_predictions, axis=0)
        final_actuals = np.concatenate(all_actuals, axis=0)
        logger.info(f"Final aggregated attributions shape: {final_attributions.shape}")
        # Sanity check: Ensure coordinates match final attributions
        if len(all_coordinates) != final_attributions.shape[0]:
             logger.warning(f"Mismatch between aggregated coordinates ({len(all_coordinates)}) and attributions ({final_attributions.shape[0]}). This might indicate an issue.")
             # Decide whether to proceed or exit? For now, log warning.

    except Exception as e:
        logger.error(f"Error aggregating attribution results: {e}", exc_info=True)
        sys.exit(1)

    # --- Save Results (Subtask 28.5) ---
    try:
        if config.output.save_attributions:
             logger.info("Saving final attributions...")
             save_interpretation_results(
                 output_dir=output_dir,
                 filename_prefix=filename_prefix,
                 attributions=final_attributions,
                 coordinates=all_coordinates,
                 interpret_config=config, # Pass the validated config
                 cli_args=args # Pass CLI args for metadata
             )
        else:
             logger.info("Skipping saving of raw attributions as per config.")
             
        # --- Feature Extraction ---
        if config.feature_extraction.top_k is not None or config.feature_extraction.threshold is not None:
            logger.info("Extracting important features...")
            extract_and_save_features(
                 output_dir=output_dir,
                 filename_prefix=filename_prefix,
                 attributions=final_attributions,
                 coordinates=all_coordinates,
                 feature_extraction_config=config.feature_extraction # Pass the sub-config
            )
        else:
             logger.info("Skipping feature extraction as top_k/threshold not specified.")
             
        # --- Plotting (Optional) ---
        if config.output.generate_plots:
             logger.info("Generating and saving plots...")
             generate_and_save_plots(
                 output_dir=output_dir,
                 filename_prefix=filename_prefix,
                 attributions=final_attributions,
                 predictions=final_predictions,
                 actuals=final_actuals,
                 coordinates=all_coordinates,
                 config=config, # Pass full config if needed for plots
                 secondary_predictions=secondary_predictions # Pass secondary predictions
             )
        else:
             logger.info("Skipping plot generation as per config.")
             
    except Exception as e:
        logger.error(f"Error during saving or feature extraction: {e}", exc_info=True)
        # Decide if we should exit or just warn? Exiting for now.
        sys.exit(1)

    logger.info("EpiBench interpretation finished successfully.")
    logger.info(f"Results saved in: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interpret an EpiBench model using Integrated Gradients.")
    setup_interpret_parser(parser)
    args = parser.parse_args()
    interpret_main(args) 