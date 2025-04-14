import yaml
import json
import os
import logging
from typing import Any, Dict, Optional, Type, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages loading, validation, and access of configuration files.
    
    Supports YAML and JSON formats.
    """
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initializes the ConfigManager with an empty configuration dictionary."""
        self._config: Dict[str, Any] = {}
        if config_path:
            self.load_config_file(config_path)
        # Placeholder for loading methods, validation, etc.

    def load_config_file(self, config_path: str) -> None:
        """Loads configuration from the specified file path.

        Determines file type (YAML or JSON) based on extension.

        Args:
            config_path: Path to the configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the file format is unsupported or parsing fails.
        """
        path = Path(config_path)
        if not path.is_file():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        _, file_extension = os.path.splitext(config_path)
        file_extension = file_extension.lower()

        try:
            if file_extension in ['.yaml', '.yml']:
                self._load_yaml(config_path)
                logger.info(f"Successfully loaded YAML configuration from: {config_path}")
            elif file_extension == '.json':
                self._load_json(config_path)
                logger.info(f"Successfully loaded JSON configuration from: {config_path}")
            else:
                raise ValueError(f"Unsupported configuration file format: {file_extension}. Use .yaml, .yml, or .json.")
        except Exception as e:
            logger.error(f"Failed to load or parse configuration file {config_path}: {e}", exc_info=True)
            # Re-raise specific types for better handling upstream if needed
            if isinstance(e, (yaml.YAMLError, json.JSONDecodeError)):
                 raise ValueError(f"Error parsing {config_path}: {e}") from e
            raise # Re-raise other exceptions

    def _load_yaml(self, file_path: str) -> None:
        """Loads configuration from a YAML file.
        
        Raises:
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        try:
            with open(file_path, 'r') as f:
                loaded_data = yaml.safe_load(f)
            # If the file is empty or just whitespace/comments, safe_load returns None
            self._config = loaded_data if loaded_data is not None else {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}", exc_info=True)
            raise  # Re-raise the specific error
        except Exception as e:
            # Catch other potential file reading errors
            logger.error(f"Error reading YAML file {file_path}: {e}", exc_info=True)
            raise ValueError(f"Could not read YAML file {file_path}") from e
    
    def _load_json(self, file_path: str) -> None:
        """Loads configuration from a JSON file.
        
        Raises:
            json.JSONDecodeError: If there is an error parsing the JSON file.
        """
        try:
            with open(file_path, 'r') as f:
                self._config = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}", exc_info=True)
            raise # Re-raise the specific error
        except Exception as e:
             # Catch other potential file reading errors
            logger.error(f"Error reading JSON file {file_path}: {e}", exc_info=True)
            raise ValueError(f"Could not read JSON file {file_path}") from e

    def validate(self, schema: Optional[Dict[str, Union[Type, Dict]]] = None) -> None:
        """Validates the loaded configuration against an optional schema.

        The schema is a dictionary where keys are parameter names (using dot notation
        for nesting, e.g., 'model.params.lr') and values are the expected types
        (e.g., int, str, float, list, dict).

        Args:
            schema: The validation schema dictionary.

        Raises:
            ValueError: If a required key from the schema is missing.
            TypeError: If a parameter has an unexpected type according to the schema.
        """
        if not isinstance(self._config, dict):
            raise TypeError("Loaded configuration is not a dictionary.")

        if schema is None:
            logger.info("No validation schema provided. Skipping validation.")
            return
            
        logger.info("Validating configuration against schema...")
        for key, expected_type in schema.items():
            keys = key.split('.')
            value = self._config
            current_key_path = ""
            key_found = True
            
            try:
                for i, k in enumerate(keys):
                    current_key_path += k
                    if isinstance(value, dict):
                        value = value[k]
                    else:
                        # Trying to access a key within a non-dict parent
                        raise KeyError(f"Parent is not a dictionary at '{current_key_path}'")
                    if i < len(keys) - 1:
                        current_key_path += '.'
                
                # Key found, now check type
                if not isinstance(value, expected_type):
                    raise TypeError(f"Invalid type for key '{key}'. Expected {expected_type.__name__}, got {type(value).__name__}.")
                
                # Optional: Add more specific validation based on expected_type if needed
                # e.g., check if list elements have a certain type, dict has required sub-keys, etc.

            except KeyError:
                # If any key along the path is missing
                raise ValueError(f"Required configuration key '{key}' (or part of its path '{current_key_path}') is missing.")
            except TypeError as e:
                 # Re-raise TypeErrors from isinstance check
                 raise e
            except Exception as e:
                 # Catch unexpected errors during validation
                 logger.error(f"Unexpected error validating key '{key}': {e}", exc_info=True)
                 raise ValueError(f"Error validating configuration key '{key}'.") from e
                 
        logger.info("Configuration validation successful.")

    def set_defaults(self, defaults: Dict[str, Any]) -> None:
        """Applies default values for missing parameters.

        Uses dot notation for nested keys in the defaults dictionary.
        If a key or part of its path doesn't exist in the current config,
        it will be created.

        Args:
            defaults: A dictionary where keys are parameter names (dot notation for nesting)
                      and values are the default values to set if the key is missing.
        """
        logger.info("Applying default configuration values...")
        if not isinstance(self._config, dict):
            logger.warning("Cannot set defaults because the loaded configuration is not a dictionary.")
            return
            
        for key, default_value in defaults.items():
            keys = key.split('.')
            current_level = self._config
            
            for i, k in enumerate(keys):
                is_last_key = (i == len(keys) - 1)
                
                if is_last_key:
                    # If it's the final key, set the default if it doesn't exist
                    if k not in current_level:
                         current_level[k] = default_value
                         logger.debug(f"Set default for '{key}': {default_value}")
                else:
                    # If it's a nested key, ensure the parent dict exists
                    if k not in current_level:
                        # Create the intermediate dictionary if it's missing
                        current_level[k] = {}
                        logger.debug(f"Created intermediate dict for key '{k}' in '{key}'")
                    elif not isinstance(current_level[k], dict):
                        # If the path exists but isn't a dict, we can't set a nested default.
                        logger.warning(f"Cannot set default for nested key '{key}' because '{k}' is not a dictionary.")
                        break # Stop processing this default key
                    
                    # Move to the next level
                    current_level = current_level[k]
                    
        logger.info("Finished applying default values.")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieves a top-level configuration value by key.

        Args:
            key: The configuration key to retrieve.
            default: The value to return if the key is not found.

        Returns:
            The configuration value or the default.
        """
        if not isinstance(self._config, dict):
             logger.error("Cannot get key '{key}' because configuration is not a dictionary.")
             return default
        return self._config.get(key, default)

    def get_nested(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieves a potentially nested configuration value using dot notation.
        
        Example: get_nested('training.optimizer.lr')
        
        Args:
            key: The dot-separated key string.
            default: Value to return if the key or any part of the path is not found
                     or if an intermediate value is not a dictionary.
        
        Returns:
            The nested value or the default.
        """
        if not isinstance(self._config, dict):
             logger.error("Cannot get nested key '{key}' because configuration is not a dictionary.")
             return default
             
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                     # Cannot traverse further if intermediate is not a dict
                     logger.debug(f"Cannot get nested key '{key}'. Intermediate value for '{k}' is not a dictionary.")
                     return default
            return value
        except KeyError:
            # Key not found at some level
             logger.debug(f"Nested key '{key}' not found.")
             return default
        except Exception as e:
             # Catch unexpected errors during traversal
             logger.error(f"Error accessing nested key '{key}': {e}", exc_info=True)
             return default

    @property
    def config(self) -> Dict[str, Any]:
        """Returns the current, validated configuration dictionary including defaults."""
        return self._config

    def __getitem__(self, key: str) -> Any:
        """Allows dictionary-style access to configuration values.

        Args:
            key: The configuration key.

        Returns:
            The configuration value.

        Raises:
            KeyError: If the key is not found in the configuration.
        """
        if key not in self._config:
            raise KeyError(f"Configuration key '{key}' not found.")
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Checks if a key exists in the configuration.

        Args:
            key: The configuration key.

        Returns:
            True if the key exists, False otherwise.
        """
        return key in self._config

# Example Usage (can be removed or kept for testing)
# if __name__ == '__main__':
#     setup_logger(log_level=logging.DEBUG) # Assuming setup_logger exists

#     # Create dummy config files
#     dummy_yaml = 'config_test.yaml'
#     dummy_json = 'config_test.json'
#     with open(dummy_yaml, 'w') as f:
#         yaml.dump({'model': {'type': 'CNN', 'lr': 0.001}, 'data': {'path': '/data'}}, f)
#     with open(dummy_json, 'w') as f:
#         json.dump({'optimizer': 'Adam', 'epochs': 10}, f)

#     manager = ConfigManager()
    
#     print("--- Loading YAML ---")
#     try:
#         manager.load_config(dummy_yaml)
#         print("Loaded config:", manager.config)
#         print("Get model.lr:", manager.get_nested('model.lr'))
#         print("Get data.path:", manager.get('data')['path']) # Example direct access
#         manager.validate() # Placeholder call
#         manager.set_defaults({'batch_size': 32}) # Placeholder call
#         print("Config after defaults:", manager.config)

#     except Exception as e:
#         print(f"Error loading YAML: {e}")
        
#     print("\n--- Loading JSON ---")
#     try:
#         manager.load_config(dummy_json) # This will overwrite previous config
#         print("Loaded config:", manager.config)
#         print("Get optimizer:", manager.get('optimizer'))
#         print("Get non-existent:", manager.get('model', 'default_value'))
#         print("Get nested non-existent:", manager.get_nested('training.batch', 64))
#     except Exception as e:
#         print(f"Error loading JSON: {e}")

#     # Clean up dummy files
#     os.remove(dummy_yaml)
#     os.remove(dummy_json) 