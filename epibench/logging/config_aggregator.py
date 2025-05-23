"""
EpiBench Configuration Aggregator

Collects and aggregates configuration parameters from temp_configs
directory for comprehensive logging and reproducibility.
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConfigurationAggregator:
    """
    Aggregates configuration parameters from temporary configuration files
    created during pipeline execution.
    """
    
    def __init__(self):
        """Initialize the ConfigurationAggregator."""
        self.configs = {}
        self.merged_config = {}
        
    def aggregate_configs(self, sample_output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Aggregate all configuration files from a sample's output directory.
        
        Args:
            sample_output_dir: Path to the sample's output directory
            
        Returns:
            Dictionary containing aggregated configuration data
        """
        sample_output_dir = Path(sample_output_dir)
        temp_configs_dir = sample_output_dir / "temp_configs"
        
        if not temp_configs_dir.exists():
            logger.warning(f"temp_configs directory not found: {temp_configs_dir}")
            return {
                "temp_configs_content": {},
                "effective_config": {},
                "config_files_found": []
            }
        
        # Find all configuration files
        config_files = self._find_config_files(temp_configs_dir)
        logger.info(f"Found {len(config_files)} configuration files in {temp_configs_dir}")
        
        # Parse each configuration file
        self.configs = {}
        for config_file in config_files:
            relative_path = config_file.relative_to(temp_configs_dir)
            try:
                config_data = self._parse_config_file(config_file)
                self.configs[str(relative_path)] = config_data
            except Exception as e:
                logger.error(f"Failed to parse config file {config_file}: {e}")
                self.configs[str(relative_path)] = {"error": str(e)}
        
        # Merge configurations
        self.merged_config = self._merge_configs(list(self.configs.values()))
        
        # Return aggregated data
        return {
            "temp_configs_content": self.configs,
            "effective_config": self.merged_config,
            "config_files_found": [str(f.relative_to(temp_configs_dir)) for f in config_files]
        }
    
    def _find_config_files(self, directory: Path) -> List[Path]:
        """
        Find all configuration files in the directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of configuration file paths
        """
        config_patterns = ["*.yaml", "*.yml", "*.json"]
        config_files = []
        
        for pattern in config_patterns:
            config_files.extend(directory.glob(pattern))
            # Also search subdirectories
            config_files.extend(directory.glob(f"**/{pattern}"))
        
        # Sort for consistent ordering
        return sorted(set(config_files))
    
    def _parse_config_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a configuration file based on its extension.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            Parsed configuration as dictionary
        """
        suffix = file_path.suffix.lower()
        
        try:
            with open(file_path, 'r') as f:
                if suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif suffix == '.json':
                    return json.load(f)
                else:
                    # Try to parse as YAML first, then JSON
                    content = f.read()
                    try:
                        return yaml.safe_load(content) or {}
                    except yaml.YAMLError:
                        return json.loads(content)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            raise
    
    def _merge_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries into one.
        
        For conflicting keys, later values override earlier ones.
        Lists are concatenated, dictionaries are merged recursively.
        
        Args:
            configs: List of configuration dictionaries
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config in configs:
            if isinstance(config, dict) and "error" not in config:
                self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge source dictionary into target dictionary.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    # Recursively merge dictionaries
                    self._deep_merge(target[key], value)
                elif isinstance(target[key], list) and isinstance(value, list):
                    # For lists, we'll keep the latest value (override)
                    # Alternative: concatenate lists with target[key].extend(value)
                    target[key] = value
                else:
                    # Override with new value
                    target[key] = value
            else:
                # Add new key
                target[key] = value
    
    def extract_key_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key parameters that are important for reproducibility.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary of key parameters
        """
        key_params = {}
        
        # Define paths to important parameters
        important_paths = [
            # Data parameters
            ("data.batch_size", "batch_size"),
            ("data.train_path", "train_data_path"),
            ("data.val_path", "validation_data_path"),
            ("data.test_path", "test_data_path"),
            
            # Model parameters
            ("model.name", "model_name"),
            ("model.params", "model_parameters"),
            
            # Training parameters
            ("training.epochs", "epochs"),
            ("training.optimizer", "optimizer"),
            ("training.optimizer_params.lr", "learning_rate"),
            ("training.loss_function", "loss_function"),
            ("training.device", "device"),
            
            # Output parameters
            ("output.checkpoint_dir", "checkpoint_directory"),
            ("checkpoint_dir", "checkpoint_directory"),  # Alternative path
        ]
        
        for path, param_name in important_paths:
            value = self._get_nested_value(config, path)
            if value is not None:
                key_params[param_name] = value
        
        return key_params
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """
        Get a value from a nested dictionary using dot notation.
        
        Args:
            data: Dictionary to search
            path: Dot-separated path to the value
            
        Returns:
            Value at the path or None if not found
        """
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def normalize_paths(self, config: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
        """
        Normalize relative paths in configuration to absolute paths.
        
        Args:
            config: Configuration dictionary
            base_path: Base path for resolving relative paths
            
        Returns:
            Configuration with normalized paths
        """
        def _normalize_value(value: Any) -> Any:
            if isinstance(value, str):
                # Check if it looks like a path
                if '/' in value or '\\' in value:
                    path = Path(value)
                    if not path.is_absolute():
                        # Make it absolute relative to base_path
                        abs_path = (base_path / path).resolve()
                        if abs_path.exists() or any(part.endswith(('.h5', '.yaml', '.yml', '.json', '.pth')) 
                                                    for part in abs_path.parts):
                            return str(abs_path)
                return value
            elif isinstance(value, dict):
                return {k: _normalize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_normalize_value(v) for v in value]
            else:
                return value
        
        return _normalize_value(config) 