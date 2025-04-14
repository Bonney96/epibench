import optuna
import logging
from typing import Dict, Any, Callable, Optional, Union, Tuple, Type

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming Trainer class and model are importable, adjust paths if necessary
from epibench.training.trainer import Trainer
from epibench.models.seq_cnn_regressor import SeqCNNRegressor 
# Need access to DataLoaders, potentially via run_optimization method
# from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class HPOptimizer:
    """
    Manages hyperparameter optimization using Optuna.
    Integrates with the Trainer class to run optimization trials.
    """
    def __init__(self, 
                 study_name: str = "epibench_hpo", 
                 direction: Union[str, optuna.study.StudyDirection] = "minimize", 
                 storage: Optional[str] = None, # e.g., "sqlite:///db.sqlite3"
                 sampler: Optional[optuna.samplers.BaseSampler] = None, # e.g., TPESampler()
                 pruner: Optional[optuna.pruners.BasePruner] = None,   # e.g., MedianPruner()
                 base_config: Optional[Dict[str, Any]] = None, 
                 train_loader: Optional[DataLoader] = None,
                 val_loader: Optional[DataLoader] = None,
                 device: Optional[torch.device] = None,
                 pruning_callback: Optional[Callable[[int, float], bool]] = None
                 ) -> None:
        """
        Initializes the HPOptimizer and creates an Optuna study.

        Args:
            study_name: Name for the Optuna study.
            direction: Direction of optimization ('minimize' or 'maximize').
            storage: Database URL for Optuna storage. If None, uses in-memory storage.
            sampler: Optuna sampler instance.
            pruner: Optuna pruner instance.
            base_config: Base configuration for the model.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: Device to use for training.
            pruning_callback: Callback for pruning support within Trainer.
        """
        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        self.sampler = sampler
        self.pruner = pruner

        try:
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                storage=self.storage,
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=True # Load existing study if name matches
            )
            logger.info(f"Optuna study '{self.study_name}' created/loaded successfully.")
            logger.info(f"  Direction: {self.direction}")
            logger.info(f"  Sampler: {self.study.sampler.__class__.__name__}")
            logger.info(f"  Pruner: {self.study.pruner.__class__.__name__}")
            logger.info(f"  Number of finished trials: {len(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")
        except Exception as e:
            logger.error(f"Failed to create/load Optuna study '{self.study_name}': {e}", exc_info=True)
            raise

        # Store necessary components for the objective function
        # These might be better passed to run_optimization and stored there
        self.base_config = base_config or {}
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pruning_callback = pruning_callback # Store internal callback

    # --- Placeholder methods for subsequent subtasks ---

    def define_search_space(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Defines the hyperparameter search space for a trial.
           Uses Optuna's trial object to suggest hyperparameter values.

        Args:
            trial: The Optuna trial object.

        Returns:
            A dictionary containing the suggested hyperparameters for this trial.
        """
        params = {
            # Optimizer Hyperparameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True), # Common range for Adam/AdamW
            # Consider suggesting optimizer type if desired:
            # 'optimizer_type': trial.suggest_categorical('optimizer_type', ['Adam', 'AdamW', 'SGD']), 

            # Model Architecture Hyperparameters (based on SeqCNNRegressor PRD)
            # Assuming filters per branch might be tuned, or a global filter multiplier
            # Example: Tune the number of filters in the first layer of each branch
            # Adjust range based on expected reasonable values
            'num_filters': trial.suggest_int('num_filters', 16, 128, step=16),
            
            # Example: Tune the number of units in the fully connected layers
            # Adjust range based on expected reasonable values
            'fc_units': trial.suggest_int('fc_units', 64, 512, step=64),
            
            # Dropout Rate
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            
            # Batch Normalization (Typically boolean - enable/disable)
            # Optuna can suggest categorical, but often it's simpler to test with/without BN
            # 'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            # If always using BN as per model design, don't include it here.
            
            # Training Hyperparameters (Can also be tuned if desired)
            # 'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        }
        
        logger.debug(f"Trial {trial.number}: Suggested hyperparameters: {params}")
        return params

    def _create_objective_callback(self, trial: optuna.trial.Trial) -> Callable[[int, float], None]:
        """Creates a callback function for Optuna pruning integration with Trainer."""
        def callback(epoch: int, current_val_loss: float):
            trial.report(current_val_loss, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch} with val_loss {current_val_loss:.4f}")
                raise optuna.TrialPruned()
        return callback

    def objective(self, trial: optuna.trial.Trial) -> float:
        """The objective function that Optuna will minimize/maximize.
        
        This function is called by Optuna for each trial.
        It configures, runs, and evaluates one training run.
        """
        logger.info(f"--- Starting Optuna Trial {trial.number} ---")

        if not self.train_loader or not self.val_loader:
            logger.error("Train/Val loaders not set in HPOptimizer. Cannot run objective.")
            # Return a high value for minimization or low for maximization
            return float('inf') if self.direction == "minimize" else float('-inf')
            
        try:
            # 1. Get hyperparameters for this trial
            params = self.define_search_space(trial)

            # 2. Configure components based on suggested parameters
            # Merge base config with HPO params (HPO params take precedence)
            current_config = self.base_config.copy()
            current_config.update(params) # Overwrite base config with HPO params
            
            # --- Model --- 
            # Assuming SeqCNNRegressor takes these HPO params directly or via a nested config
            # Adjust instantiation based on your model's __init__ signature
            model_config = current_config.get('model', {})
            model_config['num_filters'] = params.get('num_filters', model_config.get('num_filters', 64)) # Example
            model_config['fc_units'] = params.get('fc_units', model_config.get('fc_units', 128)) # Example
            model_config['dropout_rate'] = params.get('dropout_rate', model_config.get('dropout_rate', 0.5)) # Example
            # Assuming SeqCNNRegressor uses config dict; adjust if it takes direct args
            # TODO: Verify SeqCNNRegressor init signature
            model = SeqCNNRegressor(**model_config).to(self.device) 

            # --- Optimizer --- 
            lr = params.get('learning_rate', 1e-3) # Use .get for safety
            wd = params.get('weight_decay', 0)
            # Add logic for choosing optimizer type if suggested
            # optimizer_type = params.get('optimizer_type', 'Adam')
            # if optimizer_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            # elif optimizer_type == 'AdamW':
            #    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            # elif optimizer_type == 'SGD':
            #    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9) # Example momentum
            # else:
            #    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
                
            # --- Criterion (Loss Function) ---
            # Assuming it's fixed, otherwise could be part of HPO
            # Example: nn.MSELoss() for regression
            # TODO: Make sure criterion is appropriate (likely defined in base_config or fixed)
            criterion_name = current_config.get('training', {}).get('criterion', 'MSELoss')
            if criterion_name == 'MSELoss':
                criterion = nn.MSELoss()
            # Add other criteria if needed (e.g., BCEWithLogitsLoss for classification)
            else:
                 raise ValueError(f"Unsupported criterion: {criterion_name}")

            # --- Trainer --- 
            # Pass necessary components and the *merged* config
            trainer_config = current_config.get('trainer', {}) # Get trainer specific config if nested
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                config=trainer_config, # Pass trainer sub-config
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                # Pass pruning callback to Trainer if Trainer supports it
                pruning_callback=self._create_objective_callback(trial)
            )

            # 3. Run Training
            trainer.train() 

            # 4. Return the performance metric (e.g., best validation loss)
            # Make sure the direction matches Optuna study direction ('minimize' loss)
            result = trainer.best_val_loss 
            logger.info(f"--- Finished Optuna Trial {trial.number} --- Result (Val Loss): {result:.6f}")
            
            # Clean up GPU memory if needed
            del model, optimizer, criterion, trainer
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()
            
            return result

        except optuna.TrialPruned as e:
            # Let Optuna handle pruning exceptions
             logger.warning(f"Trial {trial.number} was pruned: {e}")
             raise e # Re-raise for Optuna
        except Exception as e:
            logger.error(f"Error during Optuna trial {trial.number}: {e}", exc_info=True)
            # Report failure to Optuna by returning a large/small value or raising an exception
            # Returning a poor value might be safer than raising an arbitrary exception
            return float('inf') if self.direction == "minimize" else float('-inf') 
            # Or: raise e # If you want Optuna to potentially stop

    def run_optimization(self, 
                         # Components needed by the objective function
                         base_config: Dict[str, Any],
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         device: Optional[torch.device] = None,
                         # Optuna parameters
                         n_trials: int = 50, 
                         timeout: Optional[int] = None,
                         n_jobs: int = 1, # Number of parallel trials
                         catch: Union[Tuple[()], Tuple[Type[Exception]], None] = ()
                         ) -> None:
        """Runs the Optuna optimization process.

        Args:
            base_config: Base configuration dictionary for model/training setup.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            device: Device to run training on.
            n_trials: The number of trials to run.
            timeout: Stop study after the given number of seconds.
            n_jobs: Number of parallel jobs. If -1, uses the number of CPUs.
            catch: Specify exceptions to catch during study execution.
        """
        # Store components needed by the objective function
        self.base_config = base_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device set to: {self.device}")

        logger.info(f"Starting Optuna optimization for {n_trials} trials...")
        logger.info(f"Timeout: {timeout} seconds" if timeout else "No timeout")
        logger.info(f"Parallel jobs (n_jobs): {n_jobs}")
        
        try:
            # The objective function is implicitly passed as self.objective
            self.study.optimize(
                self.objective, 
                n_trials=n_trials, 
                timeout=timeout,
                n_jobs=n_jobs,
                catch=catch # Catch specified exceptions without stopping the study
            )
            logger.info(f"Optimization finished. Total trials in study: {len(self.study.trials)}")
            # Log best trial info
            try:
                 logger.info(f"Best trial number: {self.study.best_trial.number}")
                 logger.info(f"Best value ({self.direction}): {self.study.best_value:.6f}")
                 logger.info(f"Best parameters: {self.study.best_params}")
            except ValueError:
                 logger.warning("No completed trials found in the study.")
                 
        except KeyboardInterrupt:
             logger.warning("Optimization stopped manually via KeyboardInterrupt.")
             # Optionally log current best results even if interrupted
             try:
                 logger.info(f"Current best value ({self.direction}): {self.study.best_value:.6f}")
                 logger.info(f"Current best parameters: {self.study.best_params}")
             except ValueError:
                 logger.info("No completed trials were found before interruption.")
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            # Consider re-raising or handling based on desired behavior
            raise

    def get_best_params(self) -> Dict[str, Any]:
        """Returns the best hyperparameters found so far by the Optuna study.
        
        Returns:
            A dictionary of the best hyperparameters.
            Returns an empty dictionary if no trials are completed.
        """
        try:
            return self.study.best_params
        except ValueError:
            logger.warning("No completed trials in the study yet. Cannot get best parameters.")
            return {}
        except Exception as e:
            logger.error(f"Error retrieving best parameters: {e}", exc_info=True)
            return {}

    def get_best_value(self) -> Optional[float]:
        """Returns the best objective function value found so far by the Optuna study.
        
        Returns:
            The best objective value (float), or None if no trials are completed.
        """
        try:
            return self.study.best_value
        except ValueError:
            logger.warning("No completed trials in the study yet. Cannot get best value.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving best value: {e}", exc_info=True)
            return None

# Example Usage (Optional, for testing - requires more setup)
# if __name__ == '__main__':
#     # Configure logging if not already done
#     # from ..utils.logging import LoggerManager
#     # LoggerManager.setup_logger(default_log_level=logging.INFO)
#
#     try:
#         hpo = HPOptimizer(study_name="my_test_study", direction="minimize")
#         print(f"Study '{hpo.study_name}' created.")
#         # In a real scenario, you would implement objective and run:
#         # hpo.run_optimization(n_trials=10)
#         # best_params = hpo.get_best_params()
#         # print("Best Params:", best_params)
#     except Exception as e:
#         print(f"Error during HPOptimizer initialization: {e}") 