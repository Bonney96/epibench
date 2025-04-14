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

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from epibench.config import ConfigManager
from epibench.utils.logging import LoggerManager
from epibench.models import models
from epibench.data.data_loader import create_dataloaders # Or a specific loader for interpretation data
from epibench.training.trainer import Trainer # For static load_model method
from epibench.interpretation import ModelInterpreter

logger = logging.getLogger(__name__)

def setup_interpret_parser(parser: argparse.ArgumentParser):
    """Adds arguments specific to the interpret command."""
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
        help="Path to the model checkpoint file (.pth) to interpret."
    )
    parser.add_argument(
        "-i", "--input-data",
        type=str,
        required=True,
        help="Path to the input dataset file (e.g., HDF5) for which to generate interpretations."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory to save interpretation results (attributions, plots, regions)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        help="Batch size for interpretation. Overrides config if provided."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Auto-detect by default
        help="Device to use ('cpu' or 'cuda'). Defaults to cuda if available."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target output index for attribution (for multi-output models). Defaults to argmax."
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of steps for Integrated Gradients approximation."
    )
    parser.add_argument(
        "--baseline",
        type=str, # Could be path to a baseline file or 'zeros'
        default='zeros',
        help="Baseline to use for Integrated Gradients. Default is zeros. (Future: support file path)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Extract top K features with highest (absolute) attribution."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Extract features with (absolute) attribution above this threshold."
    )
    parser.add_argument(
        "--no-abs-val",
        action="store_true",
        default=False,
        help="Use raw attribution scores (not absolute) for thresholding/top-k."
    )
    parser.add_argument(
        "--save-attributions",
        action="store_true",
        default=False,
        help="Save the raw attribution tensors to a file."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Disable generation of attribution visualization plots."
    )

def interpret_main(args):
    """Main function for the interpret command."""
    # --- Load Config (Subtask 13.2) ---
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
    logger.info("Starting EpiBench model interpretation...")
    logger.info(f"Interpretation arguments: {args}")
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

    # --- Load Model (Subtask 13.3) ---
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
        # Load model state only, no need for optimizer/scheduler
        model, _, _, checkpoint_info = Trainer.load_model(
            checkpoint_path=args.checkpoint, 
            model=model, 
            device=device,
            load_optimizer_state=False,
            load_scheduler_state=False
        ) 
        logger.info(f"Model loaded successfully from epoch {checkpoint_info.get('epoch', 'N/A')}.")
        model.eval() # IMPORTANT: Ensure model is in evaluation mode for interpretation

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

    # --- Load Input Data (Subtask 13.3) ---
    interpret_loader = None
    try:
        interpret_data_path = args.input_data # Use the path provided via CLI argument
        if not interpret_data_path:
            # Fallback to config if needed, but CLI should be primary for interpret data
            interpret_data_path = config.get('data', {}).get('interpret_path') 
            if not interpret_data_path:
                 raise ValueError("Interpretation input data path not specified via --input-data argument.")
        
        # Determine batch size (CLI arg overrides config)
        batch_size = args.batch_size or config.get('data', {}).get('batch_size', 32) 
        
        # Create a config structure for create_dataloaders
        data_config = config.get('data', {}).copy()
        data_config['batch_size'] = batch_size
        # Use a specific key for interpretation data path
        data_config['interpret_path'] = interpret_data_path 
        data_config['shuffle_interpret'] = False # Don't shuffle interpretation data
        
        logger.info(f"Loading interpretation input data from: {interpret_data_path} with batch size: {batch_size}")

        # Use create_dataloaders, assuming it can handle an 'interpret' split
        # This might require modification of create_dataloaders later
        # It should ideally return inputs ONLY, or inputs and dummy targets
        _, _, interpret_loader = create_dataloaders(data_config, splits=['interpret']) 

        if interpret_loader is None:
             raise RuntimeError("create_dataloaders did not return an interpretation loader for the 'interpret' split.")

        logger.info("Interpretation input data loaded successfully.")

    except FileNotFoundError:
        logger.error(f"Interpretation input data file not found: {interpret_data_path}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error during data loading: {e}", exc_info=True)
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Data loading runtime error: {e}", exc_info=True)
        logger.error("This might indicate that create_dataloaders needs adjustment for interpretation mode.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading interpretation input data: {e}", exc_info=True)
        sys.exit(1)

    # --- Initialize Interpreter (Subtask 13.4) ---
    interpreter = None
    try:
        logger.info("Initializing ModelInterpreter...")
        # Ensure model is correctly loaded and on the right device before passing
        if model is None:
             raise RuntimeError("Model object is None, cannot initialize interpreter.")
        
        interpreter = ModelInterpreter(model, device=device)
        logger.info("ModelInterpreter initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing ModelInterpreter: {e}", exc_info=True)
        sys.exit(1)

    # --- Run Interpretation Loop (Subtask 13.5) ---
    logger.info("Starting interpretation loop...")
    all_attributions = []
    all_inputs = [] # Store inputs for context/visualization if needed
    # all_ids = []    # Store sample IDs if available from loader

    try:
        # Determine baseline - currently only supports 'zero'
        # Future: implement 'random', 'mean', or load from file based on args.baseline
        baseline_tensor = None # Will be generated per batch if needed
        if args.baseline != 'zero':
            logger.warning(f"Baseline type '{args.baseline}' is not fully supported yet. Using zero baseline.")
            # Implement other baseline logic here if needed
        
        batch_count = 0
        # Wrap loader with tqdm for progress bar
        for batch in tqdm(interpret_loader, desc="Interpreting batches"):
            batch_count += 1
            # --- Prepare Batch Data ---
            # Assuming loader yields (inputs) or (inputs, targets) or (inputs, targets, ids)
            # We only strictly need inputs for interpretation
            if isinstance(batch, (list, tuple)):
                inputs = batch[0] 
                # ids = batch[2] if len(batch) > 2 else None # Example ID handling
            elif isinstance(batch, torch.Tensor):
                inputs = batch
                # ids = None
            else:
                 logger.warning(f"Unexpected batch type in interpretation loader: {type(batch)}. Skipping batch.")
                 continue
                 
            inputs = inputs.to(device)
            
            # --- Prepare Baselines --- 
            # Create zero baseline matching the input batch shape
            if args.baseline == 'zero':
                baseline_tensor = torch.zeros_like(inputs)
            # Add other baseline generation logic here if needed
            else:
                 baseline_tensor = torch.zeros_like(inputs) # Fallback for now

            # --- Calculate Attributions --- 
            if interpreter:
                attributions = interpreter.calculate_attributions(
                    inputs=inputs,
                    baselines=baseline_tensor,
                    target=args.target, # Use the specified target index
                    n_steps=args.n_steps
                )

                if attributions is not None:
                    # Move results to CPU and store as numpy arrays
                    all_attributions.append(attributions.cpu().numpy())
                    all_inputs.append(inputs.cpu().numpy())
                    # if ids is not None:
                    #     all_ids.extend(ids)
                else:
                    logger.warning(f"Attribution calculation returned None for batch {batch_count}. Skipping.")
            else:
                logger.error("Interpreter is not initialized. Cannot calculate attributions.")
                break # Exit loop if interpreter failed to initialize

        logger.info(f"Interpretation loop completed over {batch_count} batches.")

    except Exception as e:
        logger.error(f"Error during interpretation loop: {e}", exc_info=True)
        sys.exit(1)

    # --- Process and Save Results (Subtask 13.6) ---
    logger.info("Processing and saving interpretation results...")

    if not all_attributions:
        logger.warning("No attributions were generated. Skipping result processing and saving.")
    else:
        try:
            # Concatenate results from all batches
            logger.debug(f"Concatenating attributions from {len(all_attributions)} batches.")
            all_attributions_np = np.concatenate(all_attributions, axis=0)
            all_inputs_np = np.concatenate(all_inputs, axis=0) # Concatenate inputs as well
            logger.info(f"Aggregated attributions shape: {all_attributions_np.shape}")
            logger.info(f"Aggregated inputs shape: {all_inputs_np.shape}")

            # Save raw attributions if requested
            if args.save_attributions:
                attr_file_path = os.path.join(args.output_dir, 'attributions.npy')
                np.save(attr_file_path, all_attributions_np)
                logger.info(f"Raw attributions saved to: {attr_file_path}")

            # Extract and save high-attribution regions if requested
            if args.top_k or args.threshold:
                logger.info(f"Extracting high attribution regions (top_k={args.top_k}, threshold={args.threshold}, abs_val={not args.no_abs_val})...")
                if interpreter:
                    # Pass the concatenated numpy array to the method (assuming it handles numpy)
                    # Or convert back to tensor if the method expects tensor
                    # Modify extract_high_attribution_regions if it only accepts tensors
                    attributions_tensor_for_extraction = torch.from_numpy(all_attributions_np).to(device) # Example conversion if needed
                    
                    high_attr_indices = interpreter.extract_high_attribution_regions(
                        # attributions=all_attributions_np, # If method accepts numpy
                        attributions=attributions_tensor_for_extraction, # If method expects tensor
                        threshold=args.threshold,
                        top_k=args.top_k,
                        abs_val=not args.no_abs_val # Use flag directly
                    )
                    
                    if high_attr_indices is not None:
                        indices_file_path = os.path.join(args.output_dir, 'high_attribution_indices.npy')
                        np.save(indices_file_path, high_attr_indices)
                        logger.info(f"High attribution indices saved to: {indices_file_path} (Shape: {high_attr_indices.shape})")
                    else:
                        logger.warning("Extraction of high attribution regions failed or returned None.")
                else:
                    logger.error("Interpreter not available for extracting regions.")

            # Generate plots if not disabled
            if not args.no_plots:
                logger.info("Generating visualization plots (Placeholder)...")
                # --- Plotting Logic (Placeholder) ---
                # This section would involve using interpreter.visualize_attributions
                # or custom matplotlib code tailored to the data type (e.g., sequence data).
                # Example: Plot average attribution over sequence length, or heatmap for specific samples.
                # Need to handle potentially large data (e.g., plot first N samples or aggregate).
                # fig, axes = interpreter.visualize_attributions(
                #     attributions=torch.from_numpy(all_attributions_np[:1]).to(device), # Example: visualize first sample
                #     inputs=torch.from_numpy(all_inputs_np[:1]).to(device),
                #     method='heat_map', 
                #     sign='absolute_value',
                #     title='Sample Attribution Heatmap'
                # )
                # if fig:
                #     plot_path = os.path.join(args.output_dir, 'sample_attribution_plot.png')
                #     fig.savefig(plot_path)
                #     logger.info(f"Sample plot saved to: {plot_path}")
                #     plt.close(fig)
                pass # Placeholder for actual plotting implementation

        except Exception as e:
            logger.error(f"Error processing or saving results: {e}", exc_info=True)
            sys.exit(1)

    logger.info("EpiBench interpretation finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interpret an EpiBench model using Integrated Gradients.")
    setup_interpret_parser(parser)
    args = parser.parse_args()
    interpret_main(args) 