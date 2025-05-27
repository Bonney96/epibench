import argparse
import logging
import sys
import os
import torch.optim as optim
import copy # Import the copy module

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
        train_loader, val_loader, test_loader = create_dataloaders(config)
        logger.info("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        sys.exit(1)

    # --- Determine Execution Mode (Standard Training vs. HPO) ---
    if args.hpo:
        # --- Hyperparameter Optimization ---
        logger.info("Starting Hyperparameter Optimization (HPO)...")
        try:
            # Instantiate HPOptimizer
            # Assumes HPO config is under 'hpo' key in the main config file
            hpo_config = config.get('hpo', {})
            if not hpo_config:
                logger.error("HPO configuration ('hpo' key) not found in the config file.")
                sys.exit(1)

            # optimizer_metric = hpo_config.get('metric', 'val_loss') # Metric is implicitly val_loss in HPOptimizer.objective
            n_trials = hpo_config.get('n_trials', 20) # Default trials from HPOptimizer or override from config

            hpo_optimizer = HPOptimizer(
                base_config=config, # Pass the full config dictionary
                study_name=hpo_config.get('study_name', 'epibench_hpo_study'), # Get from HPO config or default
                direction=hpo_config.get('direction', 'minimize'),     # Get from HPO config or default
                storage=hpo_config.get('storage', None),               # Get from HPO config or default (None for in-memory)
                sampler_name=hpo_config.get('sampler', 'TPE'),       # Pass sampler name string
                pruner_name=hpo_config.get('pruner', 'MedianPruner'),       # Pass pruner name string
                train_loader=train_loader,
                val_loader=val_loader,
                device=device
            )

            # Run optimization
            hpo_optimizer.run_optimization( # HPOptimizer has its own objective method
                 n_trials=n_trials, # Pass n_trials from config
                 # timeout can be added if needed
                 # n_jobs can be added if needed
            )
            
            best_trial_params = hpo_optimizer.get_best_params()
            best_trial_value = hpo_optimizer.get_best_value()


            logger.info("HPO finished.")
            # logger.info(f"Best trial number: {best_trial.number}") # Accessing best_trial directly might not be available
            logger.info(f"Best parameters: {best_trial_params}")
            logger.info(f"Best value: {best_trial_value}")

            # After HPO, train the final model with best parameters and 50 epochs
            logger.info("Training final model with best hyperparameters...")
            
            # Create a new config for the final model, updating with best HPO params
            final_model_config = copy.deepcopy(config_manager.config) # Use copy.deepcopy
            
            # Update specific parameters from HPO results
            # This needs careful mapping from hpo_optimizer.get_best_params() keys to config structure
            if 'learning_rate' in best_trial_params:
                final_model_config['training']['optimizer_params']['lr'] = best_trial_params['learning_rate']
            if 'weight_decay' in best_trial_params:
                final_model_config['training']['optimizer_params']['weight_decay'] = best_trial_params['weight_decay']
            if 'dropout_rate' in best_trial_params:
                final_model_config['model']['params']['dropout_rate'] = best_trial_params['dropout_rate']
            # Example for fc_units - assumes HPO tunes only the first FC layer
            # And that best_params returns a flat key like 'fc_units_0' if you named it so in define_search_space
            # Or if define_search_space returned 'fc_units' as a single int for the first layer:
            if 'fc_units' in best_trial_params: # If 'fc_units' was tuned as a single value for the first layer
                 if isinstance(final_model_config['model']['params']['fc_units'], list) and \
                    len(final_model_config['model']['params']['fc_units']) > 0:
                    final_model_config['model']['params']['fc_units'][0] = best_trial_params['fc_units']
                 else: # Or if fc_units in config is just an int (less likely based on current config)
                    final_model_config['model']['params']['fc_units'] = [best_trial_params['fc_units'], 512] # Assuming a structure
            if 'num_filters' in best_trial_params:
                 final_model_config['model']['params']['num_filters'] = best_trial_params['num_filters']


            # Set epochs for the final training run
            final_model_config['training']['epochs'] = 50  # As requested
            final_model_config['training']['early_stopping_patience'] = 7 # A reasonable patience for final run

            # Update checkpoint_dir for the final model to distinguish from HPO runs
            base_hpo_checkpoint_dir = final_model_config.get('output', {}).get('checkpoint_dir', './training_results/AML_263578_SeqCNNRegressor_new_hpo')
            final_model_config['output']['checkpoint_dir'] = os.path.join(os.path.dirname(base_hpo_checkpoint_dir), "final_model")
            os.makedirs(final_model_config['output']['checkpoint_dir'], exist_ok=True)

            logger.info(f"Final model configuration: {final_model_config}")

            # Create Model with best HPO params
            model_name_final = final_model_config['model']['name']
            model_params_final = final_model_config['model'].get('params', {})
            ModelClass_final = models.get_model(model_name_final)
            model_final = ModelClass_final(**model_params_final).to(device)

            # Create Optimizer with best HPO params
            optimizer_name_final = final_model_config['training']['optimizer']
            optimizer_params_final = final_model_config['training'].get('optimizer_params', {})
            OptimizerClass_final = getattr(optim, optimizer_name_final)
            optimizer_final = OptimizerClass_final(model_final.parameters(), **optimizer_params_final)
            
            # Criterion remains the same
            loss_name_final = final_model_config['training']['loss_function']
            CriterionClass_final = getattr(torch.nn, loss_name_final)
            criterion_final = CriterionClass_final()

            # Create Trainer for the final model
            trainer_final = Trainer(
                model=model_final,
                optimizer=optimizer_final,
                criterion=criterion_final,
                train_loader=train_loader, # Use the same data loaders
                val_loader=val_loader,
                config=final_model_config, # Pass the updated config
                device=device
            )
            logger.info("Starting final model training...")
            trainer_final.train()
            logger.info("Final model training completed.")


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

            # 2. Create Optimizer and Criterion
            logger.info("Creating optimizer and criterion...")
            optimizer_name = config['training']['optimizer']
            optimizer_params = config['training'].get('optimizer_params', {})
            loss_name = config['training']['loss_function']
            # Add more robust error handling for missing keys if needed

            # Get optimizer class
            OptimizerClass = getattr(optim, optimizer_name, None)
            if OptimizerClass is None:
                raise ValueError(f"Unknown optimizer name: {optimizer_name}")
            optimizer = OptimizerClass(model.parameters(), **optimizer_params)
            logger.info(f"Optimizer '{optimizer_name}' created with params: {optimizer_params}")

            # Get loss function class
            CriterionClass = getattr(torch.nn, loss_name, None)
            if CriterionClass is None:
                raise ValueError(f"Unknown loss function name: {loss_name}")
            criterion = CriterionClass()
            logger.info(f"Criterion '{loss_name}' created.")

            # 3. Create Trainer
            logger.info("Initializing Trainer...")
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config, # Pass the main config
                device=device
                # No 'trial' object needed for standard training
            )
            logger.info("Trainer initialized.")

            # 4. Call trainer.train()
            logger.info("Starting training loop...")
            # Assuming trainer.train handles epochs, steps, logging, etc., based on config
            history = trainer.train() # Removed train_loader and val_loader arguments
            logger.info("Training loop finished.")
            logger.info(f"Training history: {history}") # Log metrics history

            # 5. Evaluate final model (optional step, can be part of trainer.train)
            logger.info("Evaluating final model...")
            # final_metrics = trainer.evaluate(val_loader)
            # logger.info(f"Final validation metrics: {final_metrics}")
            # Placeholder - evaluation might be integrated into the train loop

            # 6. Save model/results (optional step)
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