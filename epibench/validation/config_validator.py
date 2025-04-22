import yaml
from pydantic import BaseModel, Field, validator, FilePath, DirectoryPath, field_validator
from pydantic_core.core_schema import ValidationInfo
from typing import List, Optional, Literal
import logging
import os

# Define allowed logging levels
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class InputPaths(BaseModel):
    reference_genome: FilePath
    methylation_bed: FilePath
    histone_bigwigs: List[FilePath]

    @field_validator('histone_bigwigs')
    def check_histone_bigwigs_not_empty(cls, v):
        if not v:
            raise ValueError("histone_bigwigs list cannot be empty")
        return v

class ProcessingParams(BaseModel):
    window_size: int = Field(gt=0)
    step_size: int = Field(gt=0)
    target_sequence_length: int = Field(gt=0)
    methylation_bed_column: Optional[int] = Field(default=5, ge=0) # Default to 6th column (0-indexed 5)

class SplitRatios(BaseModel):
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    # test ratio is implicit

    @field_validator('validation')
    def check_ratios_sum(cls, v: float, info: ValidationInfo):
        train_ratio = info.data.get('train') 
        if train_ratio is not None and (train_ratio + v) >= 1.0:
            raise ValueError("Sum of train and validation ratios must be less than 1.0")
        return v

class LoggingConfig(BaseModel):
    level: LogLevel = Field(default="INFO")
    file: Optional[str] = None # We'll validate this path relative to output dir later if needed

class ProcessConfig(BaseModel):
    input_paths: InputPaths
    processing_params: ProcessingParams
    split_ratios: SplitRatios
    random_seed: Optional[int] = None
    logging_config: LoggingConfig = Field(default_factory=LoggingConfig)

def validate_process_config(config_path: str) -> ProcessConfig:
    """
    Loads and validates a process configuration YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A validated ProcessConfig object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValidationError: If the configuration is invalid according to the schema or keys are missing.
        yaml.YAMLError: If the file is not valid YAML.
        KeyError: If required keys are missing in the YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict):
             raise ValueError("YAML content is not a dictionary.")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")

    # Manually structure the data from the flat YAML into the nested Pydantic model structure
    try:
        model_input_data = {
            'input_paths': {
                'reference_genome': config_data['reference_genome'],
                'methylation_bed': config_data['methylation_bed'],
                'histone_bigwigs': config_data['histone_bigwigs']
            },
            'processing_params': {
                'window_size': config_data['window_size'],
                'step_size': config_data['step_size'],
                'target_sequence_length': config_data['target_sequence_length'],
                # Use .get for optional key with default from Pydantic model
                'methylation_bed_column': config_data.get('methylation_bed_column') 
            },
            'split_ratios': config_data['split_ratios'], # Assumes this is already nested correctly
            'random_seed': config_data.get('random_seed'), # Optional key
            'logging_config': config_data.get('logging', {}) # Use .get for optional section
        }
        # Remove None value for methylation_bed_column if key wasn't present, so Pydantic uses its default
        if model_input_data['processing_params']['methylation_bed_column'] is None:
            del model_input_data['processing_params']['methylation_bed_column']
            
    except KeyError as e:
        raise KeyError(f"Missing required key in configuration file {config_path}: {e}")

    # Validate the structured data using the Pydantic model
    validated_config = ProcessConfig(**model_input_data)

    # Add further validation if needed (e.g., checking output paths relative to a base dir)

    logging.info(f"Successfully validated configuration file: {config_path}")
    return validated_config

# Example Usage (can be removed or placed in a test file)
if __name__ == '__main__':
    # Create a dummy valid config for testing Pydantic models directly
    # Replace with actual path validation later
    dummy_config_path = 'path/to/your/example_config.yaml' # Replace with a real path for testing

    # Create dummy files for FilePath validation if running this directly
    # Example:
    # Path('dummy_ref.fa').touch()
    # Path('dummy_meth.bed').touch()
    # Path('dummy_hist1.bw').touch()
    
    try:
        # Note: For FilePath validation to work, the files must actually exist.
        # You might need to adjust paths or create dummy files for standalone testing.
        print(f"Attempting to validate: {dummy_config_path}")
        # config = validate_process_config(dummy_config_path) 
        # print("Validation Successful!")
        # print(config.json(indent=2))
        print("Example usage needs a valid config path and existing input files to run.")

    except (FileNotFoundError, yaml.YAMLError, Exception) as e: # Catch Pydantic's ValidationError too
        print(f"Validation Failed: {e}") 