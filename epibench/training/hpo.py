import optuna
import logging
from typing import Dict, Any, Callable, Optional, Union, Tuple, Type

import torch
# import torch.nn as nn # Not directly used in HPOptimizer for model/criterion creation
# import torch.optim as optim # Optimizer created within objective
from torch.utils.data import DataLoader

from epibench.training.trainer import Trainer
# Model is created within the objective function based on config
# from epibench.models.seq_cnn_regressor import SeqCNNRegressor 
from epibench.models import get_model # For instantiating model by name
import torch.optim as optim # For instantiating optimizer by name
import torch.nn as nn # For instantiating criterion by name


logger = logging.getLogger(__name__)

# Helper to get Optuna sampler by name
def get_optuna_sampler(sampler_name: Optional[str] = None) -> Optional[optuna.samplers.BaseSampler]:
    if sampler_name is None:
        return None # Optuna will use its default (TPE)
    sampler_name_lower = sampler_name.lower()
    if sampler_name_lower == 'tpe':
        return optuna.samplers.TPESampler()
    elif sampler_name_lower == 'random':
        return optuna.samplers.RandomSampler()
    # Add other samplers as needed, e.g., CMAESampler
    # elif sampler_name_lower == 'cmaes':
    #     return optuna.samplers.CmaEsSampler()
    else:
        logger.warning(f"Unknown sampler name: {sampler_name}. Using Optuna default.")
        return None

# Helper to get Optuna pruner by name
def get_optuna_pruner(pruner_name: Optional[str] = None) -> Optional[optuna.pruners.BasePruner]:
    if pruner_name is None:
        return None # Optuna will use its default (Median)
    pruner_name_lower = pruner_name.lower()
    if pruner_name_lower == 'median':
        return optuna.pruners.MedianPruner()
    elif pruner_name_lower == 'hyperband':
        return optuna.pruners.HyperbandPruner()
    elif pruner_name_lower == 'nop':
        return optuna.pruners.NopPruner()
    # Add other pruners as needed
    else:
        logger.warning(f"Unknown pruner name: {pruner_name}. Using Optuna default.")
        return None

class HPOptimizer:
    """
    Manages hyperparameter optimization using Optuna.
    Integrates with the Trainer class to run optimization trials.
    """
    def __init__(self, 
                 base_config: Dict[str, Any], # Full configuration dictionary
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 study_name: str = "epibench_hpo_study", 
                 direction: Union[str, optuna.study.StudyDirection] = "minimize", 
                 storage: Optional[str] = None,
                 sampler_name: Optional[str] = None, # Name of the sampler
                 pruner_name: Optional[str] = None,   # Name of the pruner
                 # pruning_callback is handled internally via Trainer now
                 ) -> None:
        """
        Initializes the HPOptimizer and creates an Optuna study.

        Args:
            base_config: Full configuration dictionary.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: Device to use for training.
            study_name: Name for the Optuna study.
            direction: Direction of optimization ('minimize' or 'maximize').
            storage: Database URL for Optuna storage. If None, uses in-memory storage.
            sampler_name: Name of the Optuna sampler to use (e.g., 'TPE', 'Random').
            pruner_name: Name of the Optuna pruner to use (e.g., 'MedianPruner', 'NopPruner').
        """
        self.base_config = base_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.study_name = study_name
        self.direction = direction
        self.storage = storage
        
        self.sampler = get_optuna_sampler(sampler_name)
        self.pruner = get_optuna_pruner(pruner_name)

        try:
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                storage=self.storage,
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=True
            )
            logger.info(f"Optuna study '{self.study_name}' created/loaded successfully.")
            logger.info(f"  Direction: {self.direction}")
            logger.info(f"  Sampler: {self.study.sampler.__class__.__name__}")
            logger.info(f"  Pruner: {self.study.pruner.__class__.__name__}")
            logger.info(f"  Number of finished trials: {len(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")
        except Exception as e:
            logger.error(f"Failed to create/load Optuna study '{self.study_name}': {e}", exc_info=True)
            raise

    def _recursive_update(self, d: Dict, u: Dict) -> Dict:
        """Recursively update dictionary d with u."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _get_value_from_path(self, dct: Dict, path: str, default: Any = None) -> Any:
        keys = path.split('.')
        for key in keys:
            try:
                dct = dct[key]
            except (KeyError, TypeError):
                return default
        return dct

    def _set_value_from_path(self, dct: Dict, path: str, value: Any):
        keys = path.split('.')
        current_level = dct
        for i, key in enumerate(keys[:-1]):
            is_last_intermediate_key = (i == len(keys) - 2)
            next_key_is_digit = keys[i+1].isdigit()

            if isinstance(current_level, list):
                try:
                    idx = int(key)
                    # Ensure list is long enough if the next level needs to be accessed
                    if idx >= len(current_level):
                        # Cannot access or create elements beyond the current list length implicitly here
                        raise IndexError(f"Index {idx} out of bounds for list at path segment {'.'.join(keys[:i+1])}")

                    # If we are about to set the final value and the target is a list
                    if is_last_intermediate_key:
                         # We will handle setting the list value in the final step
                         current_level = current_level[idx]
                         continue # Go to final setting step
                    
                    # If navigating deeper, ensure the list element is a container (dict or list)
                    if not isinstance(current_level[idx], (dict, list)):
                         # If next key suggests a list index, create a list, else dict
                         if next_key_is_digit:
                              current_level[idx] = []
                         else:
                              current_level[idx] = {}
                    current_level = current_level[idx]
                
                except ValueError:
                    raise KeyError(f"Error navigating path '{path}': Cannot use non-integer key '{key}' for list access at path segment {'.'.join(keys[:i+1])}")
                except IndexError as e:
                     raise IndexError(f"Error navigating path '{path}': {e}")

            elif isinstance(current_level, dict):
                if key not in current_level:
                    # If next key looks like an integer index, create a list, else dict
                    if next_key_is_digit:
                         current_level[key] = []
                    else:
                         current_level[key] = {}
                current_level = current_level[key]
            else:
                # Cannot navigate into a non-dict/non-list element
                raise TypeError(f"Cannot navigate path '{path}': element at path segment {'.'.join(keys[:i])} is not a dictionary or list.")

        # Set the final value
        final_key = keys[-1]
        if isinstance(current_level, list):
            try:
                idx = int(final_key)
                # Ensure list is long enough before setting
                while len(current_level) <= idx:
                     current_level.append(None) # Pad with None if necessary
                current_level[idx] = value
            except ValueError:
                 raise KeyError(f"Error setting value for path '{path}': Final key '{final_key}' is not a valid integer index for the target list.")
            except IndexError:
                 # This case should theoretically be handled by padding, but added for safety
                 raise IndexError(f"Error setting value for path '{path}': Index {idx} out of bounds after padding.")
        elif isinstance(current_level, dict):
            current_level[final_key] = value
        else:
             raise TypeError(f"Cannot set value for path '{path}': Target container is not a dictionary or list (it is {type(current_level)}). Path segment: {'.'.join(keys[:-1]) if keys else 'root'}")

    def define_search_space(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Defines the hyperparameter search space for a trial based on base_config."""
        hpo_search_config = self.base_config.get('hpo', {}).get('search_space', {})
        if not hpo_search_config:
            logger.warning("No 'hpo.search_space' found in config. Using empty params.")
            return {}

        suggested_params = {}
        for param_key, config_val in hpo_search_config.items():
            # param_key is like "training.optimizer_params.lr" or "model.params.dropout_rate"
            if not isinstance(config_val, list) or len(config_val) < 2:  # Need at least min and max
                logger.warning(f"Skipping invalid search space config for '{param_key}': {config_val}")
                continue

            param_type = "float"  # Default
            is_log = False
            step = None
            
            # Check for explicit type hint if present
            if isinstance(config_val[0], str) and ':' in config_val[0]:
                parts = config_val[0].split(':')
                hint = parts[0].lower()
                if hint == "log":
                    is_log = True
                    param_type = "float"
                    values = [float(p) for p in parts[1:]]
                elif hint == "int":
                    param_type = "int"
                    values = [int(p) for p in parts[1:]]
                    if len(values) == 3:  # min, max, step
                        step = values[2]
                        values = values[:2]  # keep only min, max
                elif hint == "cat":
                    param_type = "categorical"
                    values = parts[1:]  # The rest are choices
                else:
                    param_type = hint
                    values = config_val[1:] if len(config_val) > 1 else []
            else:
                # Determine type from values
                if all(isinstance(v, int) for v in config_val):
                    param_type = "int"
                    values = [int(v) for v in config_val]
                elif all(isinstance(v, float) for v in config_val):
                    param_type = "float"
                    values = [float(v) for v in config_val]
                elif all(isinstance(v, str) for v in config_val):
                    param_type = "categorical"
                    values = config_val
                else:
                    param_type = "float"
                    try:
                        values = [float(v) for v in config_val]
                    except ValueError:
                        logger.warning(f"Could not parse values for '{param_key}' as float: {config_val}. Skipping.")
                        continue

            # Ensure min <= max for numeric parameters
            if param_type in ["int", "float"] and len(values) >= 2:
                min_val, max_val = values[0], values[1]
                if min_val > max_val:
                    logger.warning(f"Swapping min/max values for '{param_key}' as min > max: {min_val} > {max_val}")
                    values[0], values[1] = max_val, min_val

            # Suggest value based on type
            try:
                if param_type == "float":
                    if len(values) == 2:
                        suggested_params[param_key] = trial.suggest_float(param_key, values[0], values[1], log=is_log)
                    elif len(values) == 3 and not is_log:
                        suggested_params[param_key] = trial.suggest_float(param_key, values[0], values[1], step=values[2])
                    else:
                        logger.warning(f"Float param '{param_key}' needs 2 (min,max) or 3 (min,max,step) values. Got: {values}. Skipping.")
                elif param_type == "int":
                    if len(values) >= 2:
                        suggested_params[param_key] = trial.suggest_int(param_key, values[0], values[1], step=step if step else 1)
                    else:
                        logger.warning(f"Int param '{param_key}' needs at least 2 values (min,max). Got: {values}. Skipping.")
                elif param_type == "categorical":
                    if values:
                        suggested_params[param_key] = trial.suggest_categorical(param_key, values)
                    else:
                        logger.warning(f"Categorical param '{param_key}' needs choices. Got empty list. Skipping.")
                else:
                    logger.warning(f"Unsupported param type for '{param_key}': {param_type}. Skipping.")
            except Exception as e:
                logger.error(f"Error suggesting value for '{param_key}': {e}")
                continue

        logger.debug(f"Trial {trial.number}: Suggested HPO parameters (flat keys): {suggested_params}")
        return suggested_params

    def _create_objective_callback(self, trial: optuna.trial.Trial) -> Callable[[int, float], None]:
        """Creates a callback function for Optuna pruning integration with Trainer."""
        def callback(epoch: int, current_val_metric: float): # Changed to current_val_metric
            trial.report(current_val_metric, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch} with val_metric {current_val_metric:.4f}")
                raise optuna.TrialPruned()
        return callback

    def objective(self, trial: optuna.trial.Trial) -> float:
        """The objective function that Optuna will minimize/maximize.
        
        This function is called by Optuna for each trial.
        It configures, runs, and evaluates one training run.
        """
        logger.info(f"--- Starting Optuna Trial {trial.number} ---")
        
        trial_config = {} # Start with an empty dict
        # Deep copy of base_config to avoid modifying it across trials
        for key, value in self.base_config.items():
            if isinstance(value, dict):
                trial_config[key] = {k: v for k, v in value.items()} # Shallow copy for top-level dicts
                if isinstance(trial_config[key].get('params'), dict): # Deeper copy for model.params
                     trial_config[key]['params'] = {k_p: v_p for k_p, v_p in trial_config[key]['params'].items()}
                if isinstance(trial_config[key].get('optimizer_params'), dict): # Deeper copy for training.optimizer_params
                     trial_config[key]['optimizer_params'] = {k_op: v_op for k_op, v_op in trial_config[key]['optimizer_params'].items()}

            else:
                trial_config[key] = value
        
        # Get hyperparameters for this trial using the new define_search_space
        flat_hpo_params = self.define_search_space(trial)

        # Update trial_config with flat_hpo_params
        for path_key, value in flat_hpo_params.items():
            self._set_value_from_path(trial_config, path_key, value)
        logger.debug(f"Trial {trial.number}: Effective configuration after HPO update: {trial_config}")
            
        try:
            # --- Model ---
            model_config_dict = trial_config.get('model', {})
            model_name = model_config_dict.get('name', 'SeqCNNRegressor') # Default if not specified
            model_params_dict = model_config_dict.get('params', {})
            ModelClass = get_model(model_name)
            model = ModelClass(**model_params_dict).to(self.device)

            # --- Optimizer --- 
            optimizer_config = trial_config.get('training', {}).get('optimizer_params', {})
            optimizer_name = trial_config.get('training', {}).get('optimizer', 'AdamW')
            OptimizerClass = getattr(optim, optimizer_name)
            optimizer = OptimizerClass(model.parameters(), **optimizer_config)
                
            # --- Criterion (Loss Function) ---
            criterion_name = trial_config.get('training', {}).get('loss_function', 'MSELoss')
            CriterionClass = getattr(nn, criterion_name)
            criterion = CriterionClass()

            # --- Trainer --- 
            # Ensure 'training' sub-config is passed to Trainer if it expects it for epochs, etc.
            # Or pass individual HPO trial specific settings for epochs directly.
            trainer_effective_config = trial_config # Pass the fully resolved config for this trial
            
            pruning_cb = self._create_objective_callback(trial)

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                config=trainer_effective_config, # Pass merged config
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                pruning_callback=pruning_cb 
            )
            
            trainer.train() 
            
            # Metric for Optuna. Trainer should store the best metric achieved.
            # The direction ('minimize' or 'maximize') is set in the Optuna study.
            # Trainer.best_val_loss is suitable for 'minimize'. If maximizing R2, change this.
            metric_to_optimize = trainer.best_val_loss 
            if self.direction == "maximize" and metric_to_optimize != float('-inf'):
                 # If we are maximizing, and best_val_loss is actually a loss (lower is better),
                 # we might need to return a different metric or -best_val_loss.
                 # For now, assume best_val_loss is always the metric being optimized based on study direction.
                 # If using R2, Trainer should store best_val_r2, and HPO config should specify 'maximize'.
                 pass

            logger.info(f"--- Finished Optuna Trial {trial.number} --- Result (Metric: {metric_to_optimize:.6f})")
            
            del model, optimizer, criterion, trainer
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()
            
            return metric_to_optimize

        except optuna.TrialPruned as e:
             logger.warning(f"Trial {trial.number} was pruned: {e}")
             raise e 
        except Exception as e:
            logger.error(f"Error during Optuna trial {trial.number}: {e}", exc_info=True)
            return float('inf') if self.direction == "minimize" else float('-inf') 

    def run_optimization(self, 
                         n_trials: int = 50, 
                         timeout: Optional[int] = None,
                         n_jobs: int = 1,
                         catch: Union[Tuple[()], Tuple[Type[Exception]], None] = ()
                         ) -> None:
        """Runs the Optuna optimization process."""
        logger.info(f"Starting Optuna optimization for {n_trials} trials...")
        logger.info(f"Timeout: {timeout} seconds" if timeout else "No timeout")
        logger.info(f"Parallel jobs (n_jobs): {n_jobs}")
        
        try:
            self.study.optimize(
                self.objective, 
                n_trials=n_trials, 
                timeout=timeout,
                n_jobs=n_jobs,
                catch=catch
            )
            logger.info(f"Optimization finished. Total trials in study: {len(self.study.trials)}")
            try:
                 logger.info(f"Best trial number: {self.study.best_trial.number}")
                 logger.info(f"Best value ({self.direction}): {self.study.best_value:.6f}")
                 logger.info(f"Best parameters: {self.study.best_params}") # These are the flat keys from define_search_space
            except ValueError:
                 logger.warning("No completed trials found in the study.")
                 
        except KeyboardInterrupt:
             logger.warning("Optimization stopped manually via KeyboardInterrupt.")
             try:
                 logger.info(f"Current best value ({self.direction}): {self.study.best_value:.6f}")
                 logger.info(f"Current best parameters: {self.study.best_params}")
             except ValueError:
                 logger.info("No completed trials were found before interruption.")
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            raise

    def get_best_params(self) -> Dict[str, Any]:
        """Returns the best hyperparameters found by the Optuna study.
           These are the flat parameters as defined in define_search_space.
        """
        try:
            return self.study.best_params # Returns the flat dictionary of best HPO params
        except ValueError:
            logger.warning("No completed trials in the study yet. Cannot get best parameters.")
            return {}
        except Exception as e:
            logger.error(f"Error retrieving best parameters: {e}", exc_info=True)
            return {}

    def get_best_value(self) -> Optional[float]:
        """Returns the best objective function value found by the Optuna study."""
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