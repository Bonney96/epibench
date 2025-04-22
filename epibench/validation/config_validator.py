import yaml
from pydantic import BaseModel, Field, validator, FilePath, DirectoryPath, field_validator, conint, confloat
from pydantic_core.core_schema import ValidationInfo
from typing import List, Optional, Literal, Union, Dict, Any
import logging
import os
from pathlib import Path

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

# --- Interpretation Configuration Models (Subtask 28.3 Refactored) ---

# Nested model for Integrated Gradients specific parameters
class IntegratedGradientsParams(BaseModel):
    n_steps: conint(gt=0) = Field(
        50,
        description="Number of steps for approximation in Integrated Gradients."
    )
    baseline_type: Literal['zero', 'random', 'custom'] = Field(
        'zero',
        description="Type of baseline to use for attribution. 'zero', 'random' (Gaussian noise), or 'custom' (requires custom_baseline_path)."
    )
    custom_baseline_path: Optional[FilePath] = Field(
        None,
        description="Path to a custom baseline dataset file (e.g., .npy or .h5). Required if baseline_type is 'custom'."
    )
    target_output_index: int = Field(
        0,
        description="Index of the model's output/target to interpret (e.g., for multi-output models)."
    )

    @validator('custom_baseline_path', always=True)
    def check_custom_baseline_path(cls, v, values):
        """Validate that custom_baseline_path is provided if baseline_type is 'custom'."""
        baseline_type = values.get('baseline_type')
        if baseline_type == 'custom' and v is None:
            raise ValueError("`custom_baseline_path` is required when `baseline_type` is 'custom'.")
        if baseline_type != 'custom' and v is not None:
            logger.warning(f"custom_baseline_path ('{v}') provided but baseline_type is '{baseline_type}'. The path will be ignored.")
        return v

# Nested model for Visualization parameters
class VisualizationParams(BaseModel):
    histone_names: List[str] = Field(
        # Default list based on common usage, should be customizable
        ["H3K4me", "H3K4me3", "H3K36me3", "H3K27me3", "H3K27ac", "H3K9me3"],
        description="List of histone mark names corresponding to BigWig files, in the desired plot order."
    )
    histone_bigwig_paths: List[FilePath] = Field(
        ..., # Make this required if generate_plots is true
        description="List of file paths to the ground truth BigWig files for histone marks. Order must match histone_names."
    )
    # Add other plot customization options here if needed (e.g., colormaps, dpi)
    plot_dpi: int = Field(150, description="Resolution (dots per inch) for saved plot images.")
    max_samples_to_plot: Optional[int] = Field(
        20, # Default to plotting first 20 samples
        description="Maximum number of individual sample plots to generate. If None, plot all. Set to 0 to disable individual plots."
    )
    
    @validator('histone_bigwig_paths')
    def check_paths_match_names(cls, v, values):
        names = values.get('histone_names')
        if names and len(v) != len(names):
            raise ValueError(f"Number of histone_bigwig_paths ({len(v)}) must match number of histone_names ({len(names)})." )
        return v

# Nested model for Interpretation parameters
class InterpretationParams(BaseModel):
    method: Literal['IntegratedGradients'] = Field(
        ..., # Make method required
        description="The feature attribution method to use. Currently only 'IntegratedGradients' is supported."
    )
    # Embed the method-specific parameters
    integrated_gradients: IntegratedGradientsParams = Field(
        default_factory=IntegratedGradientsParams,
        description="Parameters specific to the Integrated Gradients method."
    )
    # Add fields for other methods here if needed in the future
    # e.g., shap_params: Optional[ShapParams] = None 

# Nested model for Feature Extraction parameters
class FeatureExtractionParams(BaseModel):
    use_absolute_value: bool = Field(
        True,
        description="Whether to use absolute attribution values for ranking/thresholding."
    )
    top_k: Optional[conint(gt=0)] = Field(
        None,
        description="Extract the top K features based on attribution score. Takes precedence over threshold if both are set."
    )
    threshold: Optional[confloat(ge=0)] = Field(
        None,
        description="Extract features with attribution scores (absolute if use_absolute_value is True) above this threshold."
    )

    @validator('threshold', always=True)
    def check_top_k_or_threshold(cls, v, values):
        """Ensure at least one of top_k or threshold is set, unless feature extraction is effectively disabled."""
        top_k = values.get('top_k')
        if top_k is None and v is None:
             # This is okay if the user doesn't intend to extract features, but maybe warn?
             logger.debug("Neither top_k nor threshold is set in feature_extraction. No features will be explicitly extracted by these criteria.")
        if top_k is not None and v is not None:
            logger.warning("Both top_k and threshold are set in feature_extraction. top_k will take precedence.")
            # Optionally force threshold to None: return None
        return v

# Nested model for Output parameters
class OutputParams(BaseModel):
    save_attributions: bool = Field(
        True,
        description="Whether to save the raw attribution scores to a file."
    )
    generate_plots: bool = Field(
        True,
        description="Whether to generate and save visualizations (if supported)."
    )
    filename_prefix: str = Field(
        "interpretation",
        description="Prefix for output filenames (e.g., attribution scores, plots)."
    )

# Top-level Interpretation Configuration Model (Refactored)
class InterpretConfig(BaseModel):
    """Pydantic model for validating the main interpretation configuration file."""
    training_config: FilePath = Field(
        ..., # Make path to original training config required
        description="Path to the configuration file used for *training* the model being interpreted."
    )
    interpretation: InterpretationParams
    feature_extraction: FeatureExtractionParams = Field(
        default_factory=FeatureExtractionParams,
        description="Parameters for extracting important features based on attributions."
    )
    output: OutputParams = Field(
        default_factory=OutputParams,
        description="Parameters controlling output generation (files, plots)."
    )
    visualization: Optional[VisualizationParams] = Field(
        default_factory=VisualizationParams, # Still create by default but can be None if plotting disabled?
        description="Parameters controlling visualization generation (BigWig paths, etc.). Needed if output.generate_plots is true."
    )
    logging_config: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Configuration for logging."
    )

    class Config:
        validate_assignment = True
        # extra = 'forbid' # Consider forbidding extra fields later

    @validator('visualization', always=True)
    def check_visualization_config_needed(cls, v, values):
        output_params = values.get('output')
        if output_params and output_params.generate_plots and v is None:
            raise ValueError("'visualization' configuration section is required when 'output.generate_plots' is true.")
        # Optional: Could also validate that bigwig paths are provided within v if v is not None
        if v and not v.histone_bigwig_paths:
             raise ValueError("'visualization.histone_bigwig_paths' must be provided when 'output.generate_plots' is true.")
        return v

# Updated validation function
def validate_interpret_config(config_path: Union[str, Path]) -> InterpretConfig:
    """Loads, parses, and validates the interpretation configuration YAML file using the refactored nested model structure.

    Args:
        config_path: Path to the interpretation configuration YAML file.

    Returns:
        A validated InterpretConfig object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        ValidationError: If the configuration data does not match the InterpretConfig schema.
        Exception: For other unexpected errors during loading/validation.
    """
    config_path = Path(config_path)
    logger.info(f"Loading and validating interpretation config from: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Interpretation configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
             raise ValueError("Configuration file is empty or invalid.")
             
        # Pydantic should handle nested validation if YAML structure matches model
        validated_config = InterpretConfig(**config_data)
        logger.info("Interpretation configuration validated successfully.")
        return validated_config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing interpretation YAML file {config_path}: {e}", exc_info=True)
        raise # Re-raise the specific YAML error
    except ValidationError as e:
        logger.error(f"Interpretation configuration validation failed for {config_path}:\n{e}", exc_info=True)
        raise # Re-raise the Pydantic validation error
    except Exception as e:
        logger.error(f"An unexpected error occurred loading/validating interpretation config {config_path}: {e}", exc_info=True)
        raise # Re-raise any other unexpected error

# --- End Interpretation Configuration Models ---

# Example export if using __all__
# __all__ = ['validate_process_config', 'ProcessConfig', 'validate_interpret_config', 'InterpretConfig', 'LoggingConfig', ... ] 