import pytest
import subprocess
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, ANY
import logging as std_logging # Use alias to avoid conflict with logger variable
import sys

# Import the class to test
from epibench.pipeline.pipeline_executor import PipelineExecutor
# Import related classes/functions needed for mocking/setup
from epibench.validation.config_validator import ProcessConfig, LoggingConfig # Assuming structure
from pydantic import ValidationError

# --- Fixtures ---

@pytest.fixture
def mock_config_data():
    """Provides raw dictionary data for a valid config."""
    return {
        'output_directory': 'results/pipeline_output', # Need this for run()
        'logging': {
            'level': 'INFO',
            'file': 'executor_test.log'
        },
        # Add other minimal required fields ProcessConfig might expect
        'reference_genome': 'ref.fa', # Example field
        'methylation_bed': 'meth.bed', # Example field
        'histone_bigwigs': ['hist1.bw'], # Example field
        'window_size': 1000, # Example field
        'step_size': 500, # Example field
        'target_sequence_length': 1000, # Example field
        'split_ratios': {'train': 0.8, 'validation': 0.1}, # Example field
    }

@pytest.fixture
def mock_validated_config(mock_config_data):
    """Provides a mocked ProcessConfig object."""
    config_mock = MagicMock(spec=ProcessConfig)
    config_mock.output_directory = mock_config_data['output_directory']
    
    logging_config_mock = MagicMock(spec=LoggingConfig)
    logging_config_mock.level = mock_config_data['logging']['level']
    logging_config_mock.file = mock_config_data['logging']['file']
    config_mock.logging_config = logging_config_mock
    
    return config_mock

@pytest.fixture
def executor_instance(tmp_path, mock_validated_config):
    """Creates a PipelineExecutor instance with real paths and minimal mocking."""
    config_file = tmp_path / "test_config.yaml"
    checkpoint_file = tmp_path / "test_checkpoint.json"
    # Create dummy config file content if needed for validation mock to work
    # config_file.write_text(yaml.dump(mock_config_data())) # Optional

    # Use patches for external interactions and logging setup
    # Patch Path.mkdir globally to prevent actual directory creation during logging setup
    with patch('epibench.pipeline.pipeline_executor.validate_process_config', return_value=mock_validated_config) as mock_validate, \
         patch('epibench.pipeline.pipeline_executor.logger'), \
         patch('epibench.pipeline.pipeline_executor.logging.FileHandler'), \
         patch('epibench.pipeline.pipeline_executor.logging.StreamHandler'), \
         patch('pathlib.Path.mkdir') as mock_mkdir: # Patch mkdir globally

        # Instantiate with real Path objects
        executor = PipelineExecutor(config_file=config_file, checkpoint_file=checkpoint_file)

        mock_validate.assert_called_once_with(str(config_file))
        # Check mkdir was called for the log file's parent directory
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        return executor

# --- Test Cases ---

def test_init_success_no_checkpoint(tmp_path, mock_validated_config):
    """Test successful initialization when checkpoint file doesn't exist."""
    config_file = tmp_path / "config.yaml"
    checkpoint_file = tmp_path / "ckpt.json"
    log_file_path = Path(mock_validated_config.logging_config.file) # Get expected log path

    with patch('epibench.pipeline.pipeline_executor.validate_process_config', return_value=mock_validated_config), \
         patch('epibench.pipeline.pipeline_executor.logger'), \
         patch('epibench.pipeline.pipeline_executor.logging.FileHandler'), \
         patch('epibench.pipeline.pipeline_executor.logging.StreamHandler'), \
         patch('pathlib.Path.mkdir') as mock_mkdir: # Mock mkdir

        executor = PipelineExecutor(config_file=config_file, checkpoint_file=checkpoint_file)

    assert executor.config == mock_validated_config
    assert executor.checkpoint_file == checkpoint_file
    assert executor.checkpoint_data == {} # Should be empty dict as file doesn't exist
    assert not checkpoint_file.exists() # Verify it wasn't created
    # Check mkdir was called for the log file's parent directory
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

def test_init_success_with_checkpoint(tmp_path, mock_validated_config):
    """Test successful initialization loading existing checkpoint data."""
    config_file = tmp_path / "config.yaml"
    checkpoint_file = tmp_path / "ckpt.json"
    checkpoint_content = {'sample1': {'status': 'completed'}}
    log_file_path = Path(mock_validated_config.logging_config.file)

    # Create the checkpoint file with content
    checkpoint_file.write_text(json.dumps(checkpoint_content))
    assert checkpoint_file.exists()

    with patch('epibench.pipeline.pipeline_executor.validate_process_config', return_value=mock_validated_config), \
         patch('epibench.pipeline.pipeline_executor.logger'), \
         patch('epibench.pipeline.pipeline_executor.logging.FileHandler'), \
         patch('epibench.pipeline.pipeline_executor.logging.StreamHandler'), \
         patch('pathlib.Path.mkdir') as mock_mkdir: # Mock mkdir

        executor = PipelineExecutor(config_file=config_file, checkpoint_file=checkpoint_file)

    assert executor.config == mock_validated_config
    assert executor.checkpoint_file == checkpoint_file
    assert executor.checkpoint_data == checkpoint_content # Should load content
    # Check mkdir was called for the log file's parent directory
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

def test_init_config_validation_error(tmp_path):
    """Test initialization failure due to config validation error."""
    config_file = tmp_path / "invalid_config.yaml"
    checkpoint_file = tmp_path / "ckpt.json"
    validation_error = ValidationError.from_exception_data(title="TestError", line_errors=[])
    
    # Mock validator to raise an error, patch module-level logger
    with patch('epibench.pipeline.pipeline_executor.validate_process_config', side_effect=validation_error): 
        with patch('epibench.pipeline.pipeline_executor.logger') as mock_logger: 
            with patch('pathlib.Path.exists', return_value=False): # For _load_checkpoint during init
                executor = PipelineExecutor(config_file=config_file, checkpoint_file=checkpoint_file)
        
    assert executor.config is None
    mock_logger.error.assert_any_call(f"Configuration validation failed: {validation_error}")
    mock_logger.error.assert_any_call("PipelineExecutor initialization failed due to configuration errors.")

def test_load_checkpoint_file_not_found(executor_instance):
    """Test _load_checkpoint when the file doesn't exist (default fixture state)."""
    # executor_instance is initialized by the fixture. 
    # The fixture uses a tmp_path checkpoint_file which doesn't exist.
    # _load_checkpoint is called during executor's __init__.

    # Verify the precondition (file doesn't exist)
    assert not executor_instance.checkpoint_file.exists() 

    # Verify the result stored after initialization
    assert executor_instance.checkpoint_data == {}

    # We don't need to patch the logger *here* or assert the log message,
    # as the relevant action happened during fixture setup. Testing the log
    # message itself could be done within the fixture if desired, or in a
    # separate test focused purely on logging.

def test_load_checkpoint_invalid_json(tmp_path, mock_validated_config):
    """Test _load_checkpoint with existing but malformed JSON file."""
    config_file = tmp_path / "config.yaml"
    checkpoint_file = tmp_path / "ckpt.json" # Use a real path
    checkpoint_file.write_text("{malformed json") # Create the malformed file

    # Patch validator and logging components like in the fixture, but run init here
    with patch('epibench.pipeline.pipeline_executor.validate_process_config', return_value=mock_validated_config), \
         patch('epibench.pipeline.pipeline_executor.logger') as mock_logger, \
         patch('epibench.pipeline.pipeline_executor.logging.FileHandler'), \
         patch('epibench.pipeline.pipeline_executor.logging.StreamHandler'), \
         patch('pathlib.Path.mkdir'): # Mock mkdir for logging setup

        executor = PipelineExecutor(config_file=config_file, checkpoint_file=checkpoint_file)

    # Check results after initialization
    assert executor.checkpoint_data == {}
    mock_logger.error.assert_any_call(f"Error loading checkpoint file {checkpoint_file}: Expecting property name enclosed in double quotes: line 1 column 2 (char 1). Starting fresh.")

def test_save_checkpoint(executor_instance):
    """Test _save_checkpoint writes correct data to the real path."""
    # executor_instance has a real checkpoint_file path (from tmp_path)
    save_path = executor_instance.checkpoint_file
    executor_instance.checkpoint_data = {'sampleA': {'status': 'failed'}, 'sampleB': {'status': 'completed'}}

    # Call the method
    executor_instance._save_checkpoint()

    # Verify the real file content
    assert save_path.exists()
    with open(save_path, 'r') as f:
        saved_data = json.load(f)
    assert saved_data == executor_instance.checkpoint_data

# --- Tests for run() method ---

@pytest.fixture
def mock_run_dependencies(executor_instance):
    """Patch dependencies needed for the run() method test."""
    # executor_instance now uses real paths, adjust mocking here accordingly
    mock_subprocess_run = MagicMock(spec=subprocess.CompletedProcess)
    mock_subprocess_run.returncode = 0
    mock_subprocess_run.stdout = "Pipeline script finished."
    mock_subprocess_run.stderr = ""

    # Mock NamedTemporaryFile - keep this mock
    mock_temp_file_handle = mock_open().return_value # Get a mock file handle
    temp_file_name = "/tmp/dummy_temp_file.yaml" # Keep a concrete name
    mock_temp_file_handle.name = temp_file_name
    mock_temp_file_obj = MagicMock()
    mock_temp_file_obj.__enter__.return_value = mock_temp_file_handle
    mock_temp_file_obj.__exit__.return_value = None

    # Now, only patch Path constructor for the *specific* paths created *inside* run()
    # The executor's own paths (config, checkpoint, output_dir from config) are real.
    output_dir_str = executor_instance.config.output_directory
    script_path_str = "scripts/run_full_pipeline.py"

    mock_temp_path_instance = MagicMock(spec=Path, name="mock_temp_path")
    mock_temp_path_instance.exists.return_value = True # Assume exists for unlink
    mock_temp_path_instance.__str__.return_value = temp_file_name
    # Mock unlink method on the specific instance
    mock_temp_path_instance.unlink = MagicMock() 

    mock_script_path_instance = MagicMock(spec=Path, name="mock_script_path")
    mock_script_path_instance.__str__.return_value = script_path_str

    # Output dir path is taken from config, which is real.
    # However, run() does `base_output_dir = Path(base_output_dir)`
    # So we need to mock the Path constructor call for *that specific string*
    mock_output_dir_instance = MagicMock(spec=Path, name="mock_output_dir")
    mock_output_dir_instance.__str__.return_value = output_dir_str

    def run_path_side_effect(arg):
        str_arg = str(arg) # Ensure we compare strings
        if str_arg == script_path_str:
            # This is the call Path("scripts/run_full_pipeline.py")
            return mock_script_path_instance
        elif str_arg == output_dir_str:
            # This is the call Path(self.config.output_directory)
            return mock_output_dir_instance
        elif str_arg == temp_file_name:
             # This is the call Path(tmp_yaml.name)
             return mock_temp_path_instance
        # Let other Path calls (if any) use the default real Path
        return Path(arg)


    # Patch module-level logger and other dependencies for run() scope
    with patch('epibench.pipeline.pipeline_executor.subprocess.run', return_value=mock_subprocess_run) as sp_run, \
         patch('epibench.pipeline.pipeline_executor.tempfile.NamedTemporaryFile', return_value=mock_temp_file_obj) as mock_tempfile_constructor, \
         patch('epibench.pipeline.pipeline_executor.Path', side_effect=run_path_side_effect) as mock_path_constructor_for_run, \
         patch('epibench.pipeline.pipeline_executor.yaml.dump') as mock_yaml_dump, \
         patch('epibench.pipeline.pipeline_executor.os.fsync'), \
         patch.object(executor_instance, '_save_checkpoint') as mock_save_ckpt, \
         patch('epibench.pipeline.pipeline_executor.logger') as mock_logger: # Patch module logger for run()

        yield {
            "executor": executor_instance,
            "subprocess_run": sp_run,
            "tempfile_constructor": mock_tempfile_constructor,
            "temp_file_handle": mock_temp_file_handle,
            "yaml_dump": mock_yaml_dump,
            "temp_path_mock": mock_temp_path_instance, # Use the specific mock instance
            "save_checkpoint": mock_save_ckpt,
            "logger": mock_logger,
            "mock_path_constructor_for_run": mock_path_constructor_for_run # Expose run's path mock
        }

def test_run_all_samples_completed(mock_run_dependencies):
    """Test run() when all samples are already marked completed in checkpoint."""
    executor = mock_run_dependencies['executor']
    logger = mock_run_dependencies['logger']
    sp_run = mock_run_dependencies['subprocess_run']
    save_checkpoint = mock_run_dependencies['save_checkpoint']

    executor.checkpoint_data = {
        'sample1': {'status': 'completed'},
        'sample2': {'status': 'completed'}
    }
    sample_list = ['sample1', 'sample2']
    sample_details = {'sample1': {}, 'sample2': {}} # Details needed but content irrelevant here

    executor.run(sample_list, sample_details)

    logger.info.assert_any_call("Sample sample1 already completed. Skipping.")
    logger.info.assert_any_call("Sample sample2 already completed. Skipping.")
    logger.info.assert_any_call("No samples need processing in this run (all completed or skipped).")
    sp_run.assert_not_called() # Subprocess should not run
    save_checkpoint.assert_not_called() # Checkpoint should not be saved if no samples processed

def test_run_some_samples_pending_success(mock_run_dependencies):
    """Test run() with pending/failed samples, successful subprocess execution."""
    executor = mock_run_dependencies['executor']
    sp_run = mock_run_dependencies['subprocess_run']
    yaml_dump = mock_run_dependencies['yaml_dump']
    save_checkpoint = mock_run_dependencies['save_checkpoint']
    temp_file_handle = mock_run_dependencies['temp_file_handle']
    temp_path_mock = mock_run_dependencies['temp_path_mock'] # Mock for the temp file path
    mock_path_constructor_for_run = mock_run_dependencies['mock_path_constructor_for_run'] # Path mock for run()

    # Set checkpoint data for this scenario
    executor.checkpoint_data = {
        'sample1': {'status': 'completed'}, # Skip
        'sample2': {'status': 'failed'},   # Run
        'sample3': {}                      # Run (no status == pending)
    }
    sample_list = ['sample1', 'sample2', 'sample3', 'sample4'] # sample4 has no details
    sample_details = {
        'sample1': {'process_data_config': 'cfg1'}, # Needed for structure
        'sample2': {'process_data_config': 'cfg2'},
        'sample3': {'process_data_config': 'cfg3'}
    }

    # Ensure subprocess mock is successful
    sp_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

    executor.run(sample_list, sample_details)

    # Verify Path constructor was called inside run for script, output dir, and temp file
    mock_path_constructor_for_run.assert_any_call("scripts/run_full_pipeline.py")
    mock_path_constructor_for_run.assert_any_call(executor.config.output_directory)
    mock_path_constructor_for_run.assert_any_call(temp_file_handle.name)

    sp_run.assert_called_once()
    args, kwargs = sp_run.call_args
    command_list = args[0]
    assert command_list[0] == sys.executable # Check python interpreter
    assert command_list[1] == "scripts/run_full_pipeline.py" # Uses __str__ of mock_script_path_instance
    assert command_list[2] == "--samples-config"
    assert command_list[3] == temp_file_handle.name # Comes from mock handle
    assert command_list[4] == "--output-dir"
    assert command_list[5] == executor.config.output_directory # Uses __str__ of mock_output_dir_instance
    assert command_list[6] == "--max-workers"
    assert command_list[7] == "1"

    yaml_dump.assert_called_once()
    dump_args, dump_kwargs = yaml_dump.call_args
    dumped_data = dump_args[0]
    assert len(dumped_data) == 2
    assert dumped_data[0]['name'] == 'sample2'
    assert dumped_data[1]['name'] == 'sample3'
    assert dump_args[1] == temp_file_handle # Check correct file handle

    save_checkpoint.assert_called_once()
    # Check final checkpoint data state
    assert executor.checkpoint_data['sample1']['status'] == 'completed' # Unchanged
    assert executor.checkpoint_data['sample2']['status'] == 'completed' # Updated
    assert executor.checkpoint_data['sample3']['status'] == 'completed' # Updated
    assert 'sample4' not in executor.checkpoint_data # Skipped

    # Check temp file cleanup was called on the specific mock path object for temp file
    temp_path_mock.unlink.assert_called_once_with()

def test_run_subprocess_fails(mock_run_dependencies):
    """Test run() when the subprocess call fails."""
    executor = mock_run_dependencies['executor']
    sp_run = mock_run_dependencies['subprocess_run']
    yaml_dump = mock_run_dependencies['yaml_dump']
    save_checkpoint = mock_run_dependencies['save_checkpoint']
    logger = mock_run_dependencies['logger']
    temp_path_mock = mock_run_dependencies['temp_path_mock']

    executor.checkpoint_data = {} 
    sample_list = ['sampleA']
    sample_details = {'sampleA': {'process_data_config': 'cfgA'}}
    
    sp_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="test cmd", stderr="Error occurred")

    executor.run(sample_list, sample_details)

    sp_run.assert_called_once()
    yaml_dump.assert_called_once()
    
    logger.error.assert_any_call("Pipeline script failed for the batch with exit code 1")
    
    save_checkpoint.assert_called_once()
    assert executor.checkpoint_data['sampleA']['status'] == 'failed'
    # Check temp file cleanup was called on the mock path object
    temp_path_mock.unlink.assert_called_once_with()

def test_run_aborts_if_config_invalid(tmp_path):
    """Test that run() does not proceed if config failed validation."""
    config_file = tmp_path / "invalid_config.yaml"
    
    with patch('epibench.pipeline.pipeline_executor.validate_process_config', side_effect=ValidationError.from_exception_data("e",[])), \
         patch('epibench.pipeline.pipeline_executor.logger') as mock_logger, \
         patch('pathlib.Path.exists', return_value=False): # For _load_checkpoint during init

        executor = PipelineExecutor(config_file=config_file)
        
        executor.run(['sample1'], {'sample1':{}})

        mock_logger.error.assert_any_call("Cannot run pipeline: Configuration is invalid or was not loaded.")

def test_run_temp_file_cleanup_on_error(mock_run_dependencies):
    """Test that temporary file is cleaned up even if subprocess fails."""
    executor = mock_run_dependencies['executor']
    sp_run = mock_run_dependencies['subprocess_run']
    temp_path_mock = mock_run_dependencies['temp_path_mock']

    executor.checkpoint_data = {} 
    sample_list = ['sampleA']
    sample_details = {'sampleA': {'process_data_config': 'cfgA'}}
    
    sp_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="test cmd")

    executor.run(sample_list, sample_details)
    
    # Ensure unlink is still called even on error
    temp_path_mock.unlink.assert_called_once_with()

# TODO: Add more tests:
# - Test logging setup variations (_setup_logging)
# - Test error during checkpoint saving/loading IOErrors
# - Test case where base_output_dir is missing in config
# - Test interaction with sample details more thoroughly (e.g., missing keys)
# - Test specific logging messages for different scenarios
