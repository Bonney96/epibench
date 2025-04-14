import argparse
import logging
import sys
import os
import torch.optim as optim

# Add project root to sys.path to allow imports from epibench
# This assumes the script is run from the project root or similar context
# A more robust approach might involve package installation
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary components (will be used in later subtasks)
from epibench.config.config_manager import ConfigManager
from epibench.data.data_loader import create_dataloaders
from epibench.models import models
from epibench.training.trainer import Trainer
from epibench.training.hpo import HPOptimizer
from epibench.utils.logging import LoggerManager
import torch

logger = logging.getLogger(__name__)

def setup_arg_parser(parser: argparse.ArgumentParser):
    """Adds arguments for the train command to the provided subparser."""
    parser.description = "Train an EpiBench model." # Set description on the passed-in parser

    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the YAML or JSON configuration file for the training run."
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        default=False,
        help="Enable Hyperparameter Optimization (HPO) using Optuna based on the 'hpo' section in the config file."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the default logging level (might be overridden by config file)."
    )
    # Add other potential arguments like --checkpoint, --device, --num-workers etc. later if needed

def main(args):
    """Main function to parse arguments and orchestrate training."""
    # The parser is now handled by the main CLI entry point
    # args = parser.parse_args() # Args are passed directly to this function now

    # --- Load Configuration First (to get logging settings) ---
    config_manager = None # Initialize to None
    try:
        config_manager = ConfigManager(args.config)
        # Basic validation or default setting can happen here if needed before logging setup
        config = config_manager.config # Get config early for logging setup
    except FileNotFoundError:
        # Use basic print/logging if config load fails before logger setup
        logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s - %(levelname)s - %(message)s')
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s - %(levelname)s - %(message)s')
        logger.error(f"Critical error loading configuration from {args.config}: {e}", exc_info=True)
        sys.exit(1)

    # --- Setup Logging using LoggerManager ---
    # Pass the loaded config_manager to setup the logger
    # LoggerManager will use defaults if logging section is missing in config
    LoggerManager.setup_logger(
        config_manager=config_manager,
        default_log_level=getattr(logging, args.log_level.upper()), # Use arg log level as default
        force_reconfigure=True # Ensure it sets up even if configured elsewhere
    )

    # Now use the configured logger
    logger.info("Starting EpiBench training process...")
    logger.info(f"Command line arguments: {args}")
    logger.info(f"Loaded configuration from: {args.config}")
    logger.debug(f"Full configuration: {config}") # Log full config at debug level

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Data ---
    logger.info("Loading data...")
    try:
        # Assuming create_dataloaders takes config and returns train/val loaders
        train_loader, val_loader = create_dataloaders(config)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        sys.exit(1)

    # --- Determine Execution Mode (Standard Training vs. HPO) ---
    if args.hpo:
        # --- Hyperparameter Optimization ---
        logger.info("Starting Hyperparameter Optimization (HPO)...")
        try:
            # Define a factory function to create Trainer instances for Optuna trials
            def trainer_factory(trial, hpo_config):
                # Sample hyperparameters using the trial object
                trial_lr = trial.suggest_float("lr", hpo_config['lr_min'], hpo_config['lr_max'], log=True)
                trial_batch_size = trial.suggest_categorical("batch_size", hpo_config['batch_sizes'])
                trial_optimizer_name = trial.suggest_categorical("optimizer", hpo_config['optimizers'])

                # Create a trial-specific config (or update a copy)
                # Note: This assumes HPO settings override base config settings
                trial_config = config.copy() # Start with base config
                trial_config['training']['learning_rate'] = trial_lr
                trial_config['data']['batch_size'] = trial_batch_size
                trial_config['training']['optimizer'] = trial_optimizer_name
                # Potentially update data loaders if batch size changes?
                # For now, assume create_dataloaders handles batch size or we recreate them.
                # trial_train_loader, trial_val_loader = create_dataloaders(trial_config)

                # Create model based on config
                model_name = trial_config['model']['name']
                model_params = trial_config['model'].get('params', {})
                # Assuming a way to get the model class/constructor
                ModelClass = models.get_model(model_name) # Needs implementation in models/__init__.py
                model = ModelClass(**model_params).to(device)

                # Instantiate Trainer for this trial
                trainer = Trainer(
                    model=model,
                    config=trial_config, # Pass trial-specific config
                    device=device,
                    trial=trial # Pass the Optuna trial object to Trainer if needed for reporting
                )
                return trainer # Return only the trainer

            # Instantiate HPOptimizer
            # Assumes HPO config is under 'hpo' key in the main config file
            hpo_config = config.get('hpo', {})
            if not hpo_config:
                logger.error("HPO configuration ('hpo' key) not found in the config file.")
                sys.exit(1)

            optimizer_metric = hpo_config.get('metric', 'val_loss') # Default metric
            n_trials = hpo_config.get('n_trials', 20) # Default trials

            hpo_optimizer = HPOptimizer(
                config_manager=config_manager, # Pass the manager or just the config dict
                metric=optimizer_metric,
                # Pass trainer_factory directly
            )

            # Run optimization
            # HPOptimizer's optimize method now takes the factory
            best_trial = hpo_optimizer.optimize(
                 trainer_factory=trainer_factory,
                 train_loader=train_loader, # Pass loaders here
                 val_loader=val_loader,
                 n_trials=n_trials
            )

            logger.info("HPO finished.")
            logger.info(f"Best trial number: {best_trial.number}")
            logger.info(f"Best parameters: {best_trial.params}")
            logger.info(f"Best value ({optimizer_metric}): {best_trial.value}")

        except ImportError as e:
             logger.error(f"Error during HPO setup: {e}. Make sure Optuna is installed (`pip install optuna`)", exc_info=True)
             sys.exit(1)
        except KeyError as e:
             logger.error(f"Missing configuration key during HPO setup: {e}", exc_info=True)
             sys.exit(1)
        except Exception as e:
            logger.error(f"An error occurred during HPO: {e}", exc_info=True)
            sys.exit(1)

    else:
        # --- Standard Training ---
        logger.info("Starting standard training...")
        try:
            # 1. Create Model
            model_name = config['model']['name']
            model_params = config['model'].get('params', {})
            logger.info(f"Creating model: {model_name} with params: {model_params}")
            ModelClass = models.get_model(model_name)
            model = ModelClass(**model_params).to(device)
            logger.info(f"Model {model_name} created successfully.")

            # 2. Create Trainer
            logger.info("Initializing Trainer...")
            trainer = Trainer(
                model=model,
                config=config, # Pass the main config
                device=device
                # No 'trial' object needed for standard training
            )
            logger.info("Trainer initialized.")

            # 3. Call trainer.train()
            logger.info("Starting training loop...")
            # Assuming trainer.train handles epochs, steps, logging, etc., based on config
            history = trainer.train(train_loader, val_loader)
            logger.info("Training loop finished.")
            logger.info(f"Training history: {history}") # Log metrics history

            # 4. Evaluate final model (optional step, can be part of trainer.train)
            logger.info("Evaluating final model...")
            # final_metrics = trainer.evaluate(val_loader)
            # logger.info(f"Final validation metrics: {final_metrics}")
            # Placeholder - evaluation might be integrated into the train loop

            # 5. Save model/results (optional step)
            logger.info("Saving model and results...")
            # trainer.save_model(output_dir=config.get('output_dir', 'results'))
            # trainer.save_results(output_dir=config.get('output_dir', 'results'))
            # Placeholder - saving logic depends on Trainer implementation

            logger.info("Standard training completed successfully.")

        except KeyError as e:
            logger.error(f"Missing configuration key during standard training setup: {e}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logger.error(f"An error occurred during standard training: {e}", exc_info=True)
            sys.exit(1)

    logger.info("EpiBench training process finished.")

if __name__ == "__main__":
    main() 