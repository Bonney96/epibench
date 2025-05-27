import pytest
import yaml
import os
from pathlib import Path
from pydantic import ValidationError

# Adjust the import path based on your project structure
from epibench.validation.config_validator import validate_process_config, ProcessConfig

# Helper to create dummy files
def create_dummy_files(base_path: Path, filenames: list):
    for fname in filenames:
        (base_path / fname).touch()

@pytest.fixture
def valid_config_dict():
    """Provides a dictionary for a valid configuration."""
    return {
        'reference_genome': 'dummy_ref.fa',
        'methylation_bed': 'dummy_meth.bed',
        'histone_bigwigs': ['dummy_hist1.bw', 'dummy_hist2.bw'],
        'window_size': 10000,
        'step_size': 5000,
        'target_sequence_length': 10000,
        'methylation_bed_column': 5,
        'split_ratios': {
            'train': 0.7,
            'validation': 0.15
        },
        'random_seed': 42,
        'logging': {
            'level': 'DEBUG',
            'file': 'processing.log'
        }
    }

@pytest.fixture
def valid_config_file(tmp_path, valid_config_dict):
    """Creates a valid config file in a temporary directory."""
    config_path = tmp_path / "valid_config.yaml"
    # Create dummy files required by FilePath validator
    dummy_files = [
        valid_config_dict['reference_genome'],
        valid_config_dict['methylation_bed'],
    ] + valid_config_dict['histone_bigwigs']
    create_dummy_files(tmp_path, dummy_files)
    
    # Update paths in dict to be absolute for the test
    valid_config_dict['reference_genome'] = str(tmp_path / valid_config_dict['reference_genome'])
    valid_config_dict['methylation_bed'] = str(tmp_path / valid_config_dict['methylation_bed'])
    valid_config_dict['histone_bigwigs'] = [str(tmp_path / p) for p in valid_config_dict['histone_bigwigs']]

    with open(config_path, 'w') as f:
        yaml.dump(valid_config_dict, f)
    return config_path

# --- Test Cases ---

def test_valid_config(valid_config_file):
    """Tests successful validation of a correct config file."""
    config = validate_process_config(str(valid_config_file))
    assert isinstance(config, ProcessConfig)
    assert config.input_paths.reference_genome.name == 'dummy_ref.fa'
    assert config.processing_params.window_size == 10000
    assert config.split_ratios.train == 0.7
    assert config.logging_config.level == 'DEBUG'
    assert config.random_seed == 42
    assert config.processing_params.methylation_bed_column == 5 # Check explicit value
    assert len(config.input_paths.histone_bigwigs) == 2

def test_valid_config_defaults(tmp_path, valid_config_dict):
    """Tests successful validation when optional keys use defaults."""
    # Remove optional keys to test defaults
    del valid_config_dict['random_seed']
    del valid_config_dict['logging']
    del valid_config_dict['methylation_bed_column']

    config_path = tmp_path / "default_config.yaml"
    # Create dummy files required by FilePath validator
    dummy_files = [
        valid_config_dict['reference_genome'],
        valid_config_dict['methylation_bed'],
    ] + valid_config_dict['histone_bigwigs']
    create_dummy_files(tmp_path, dummy_files)
    
    # Update paths in dict to be absolute
    valid_config_dict['reference_genome'] = str(tmp_path / valid_config_dict['reference_genome'])
    valid_config_dict['methylation_bed'] = str(tmp_path / valid_config_dict['methylation_bed'])
    valid_config_dict['histone_bigwigs'] = [str(tmp_path / p) for p in valid_config_dict['histone_bigwigs']]
    
    with open(config_path, 'w') as f:
        yaml.dump(valid_config_dict, f)
        
    config = validate_process_config(str(config_path))
    assert isinstance(config, ProcessConfig)
    assert config.random_seed is None # Default is None
    assert config.logging_config.level == 'INFO' # Default level
    assert config.logging_config.file is None # Default file
    assert config.processing_params.methylation_bed_column == 5 # Default column index

def test_missing_required_key(tmp_path, valid_config_dict):
    """Tests error when a required top-level key is missing."""
    del valid_config_dict['reference_genome'] # Remove a required key
    config_path = tmp_path / "missing_key_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(valid_config_dict, f)
        
    with pytest.raises(KeyError, match="'reference_genome'"):
        validate_process_config(str(config_path))

def test_missing_required_nested_key(tmp_path, valid_config_dict):
    """Tests error when a required nested key is missing."""
    del valid_config_dict['split_ratios']['train'] # Remove required nested key
    config_path = tmp_path / "missing_nested_key_config.yaml"
    # Create dummy files
    dummy_files = [valid_config_dict['reference_genome'], valid_config_dict['methylation_bed']] + valid_config_dict['histone_bigwigs']
    create_dummy_files(tmp_path, dummy_files)
    # Update paths
    valid_config_dict['reference_genome'] = str(tmp_path / valid_config_dict['reference_genome'])
    valid_config_dict['methylation_bed'] = str(tmp_path / valid_config_dict['methylation_bed'])
    valid_config_dict['histone_bigwigs'] = [str(tmp_path / p) for p in valid_config_dict['histone_bigwigs']]

    with open(config_path, 'w') as f:
        yaml.dump(valid_config_dict, f)

    with pytest.raises(ValidationError, match="split_ratios.train\n.*Field required"):
         validate_process_config(str(config_path))

def test_invalid_file_path(tmp_path, valid_config_dict):
    """Tests error when a FilePath does not exist."""
    # Don't create dummy files for this test
    config_path = tmp_path / "invalid_path_config.yaml"
    # Keep paths relative for this test to ensure non-existence
    with open(config_path, 'w') as f:
        yaml.dump(valid_config_dict, f)

    with pytest.raises(ValidationError, match="Path does not point to a file"):
        validate_process_config(str(config_path))

def test_invalid_split_ratios(tmp_path, valid_config_dict):
    """Tests error when split ratios sum >= 1.0."""
    valid_config_dict['split_ratios']['train'] = 0.8
    valid_config_dict['split_ratios']['validation'] = 0.3 # Sum is 1.1

    config_path = tmp_path / "invalid_ratio_config.yaml"
    # Create dummy files
    dummy_files = [valid_config_dict['reference_genome'], valid_config_dict['methylation_bed']] + valid_config_dict['histone_bigwigs']
    create_dummy_files(tmp_path, dummy_files)
    # Update paths
    valid_config_dict['reference_genome'] = str(tmp_path / valid_config_dict['reference_genome'])
    valid_config_dict['methylation_bed'] = str(tmp_path / valid_config_dict['methylation_bed'])
    valid_config_dict['histone_bigwigs'] = [str(tmp_path / p) for p in valid_config_dict['histone_bigwigs']]
    
    with open(config_path, 'w') as f:
        yaml.dump(valid_config_dict, f)

    with pytest.raises(ValidationError, match="Sum of train and validation ratios must be less than 1.0"):
        validate_process_config(str(config_path))

def test_invalid_logging_level(tmp_path, valid_config_dict):
    """Tests error for an invalid logging level string."""
    valid_config_dict['logging']['level'] = 'INVALID_LEVEL'

    config_path = tmp_path / "invalid_log_config.yaml"
    # Create dummy files
    dummy_files = [valid_config_dict['reference_genome'], valid_config_dict['methylation_bed']] + valid_config_dict['histone_bigwigs']
    create_dummy_files(tmp_path, dummy_files)
    # Update paths
    valid_config_dict['reference_genome'] = str(tmp_path / valid_config_dict['reference_genome'])
    valid_config_dict['methylation_bed'] = str(tmp_path / valid_config_dict['methylation_bed'])
    valid_config_dict['histone_bigwigs'] = [str(tmp_path / p) for p in valid_config_dict['histone_bigwigs']]
    
    with open(config_path, 'w') as f:
        yaml.dump(valid_config_dict, f)

    with pytest.raises(ValidationError, match="Input should be (?:\'DEBUG\'|\'INFO\'|\'WARNING\'|\'ERROR\'|\'CRITICAL\')"):
        validate_process_config(str(config_path))

def test_empty_histone_list(tmp_path, valid_config_dict):
    """Tests error when histone_bigwigs list is empty."""
    valid_config_dict['histone_bigwigs'] = []

    config_path = tmp_path / "empty_histone_config.yaml"
    # Create dummy files (excluding histones)
    dummy_files = [valid_config_dict['reference_genome'], valid_config_dict['methylation_bed']]
    create_dummy_files(tmp_path, dummy_files)
    # Update paths
    valid_config_dict['reference_genome'] = str(tmp_path / valid_config_dict['reference_genome'])
    valid_config_dict['methylation_bed'] = str(tmp_path / valid_config_dict['methylation_bed'])
    
    with open(config_path, 'w') as f:
        yaml.dump(valid_config_dict, f)

    with pytest.raises(ValidationError, match="histone_bigwigs list cannot be empty"):
        validate_process_config(str(config_path))

def test_malformed_yaml(tmp_path):
    """Tests error for invalid YAML syntax."""
    config_path = tmp_path / "malformed_config.yaml"
    with open(config_path, 'w') as f:
        f.write("key1: value1\nkey2: value2: extra_colon\n") # Invalid YAML

    with pytest.raises(yaml.YAMLError):
        validate_process_config(str(config_path))

def test_config_not_found():
    """Tests FileNotFoundError for non-existent config file."""
    with pytest.raises(FileNotFoundError):
        validate_process_config("non_existent_config_file.yaml")

def test_invalid_data_type(tmp_path, valid_config_dict):
    """Tests error when a value has the wrong data type."""
    valid_config_dict['window_size'] = "not_an_integer"

    config_path = tmp_path / "invalid_type_config.yaml"
    # Create dummy files
    dummy_files = [valid_config_dict['reference_genome'], valid_config_dict['methylation_bed']] + valid_config_dict['histone_bigwigs']
    create_dummy_files(tmp_path, dummy_files)
    # Update paths
    valid_config_dict['reference_genome'] = str(tmp_path / valid_config_dict['reference_genome'])
    valid_config_dict['methylation_bed'] = str(tmp_path / valid_config_dict['methylation_bed'])
    valid_config_dict['histone_bigwigs'] = [str(tmp_path / p) for p in valid_config_dict['histone_bigwigs']]

    with open(config_path, 'w') as f:
        yaml.dump(valid_config_dict, f)
        
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        validate_process_config(str(config_path)) 