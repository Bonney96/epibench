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
# --- Remove config imports ---
# from epibench.validation.config_validator import ProcessConfig, LoggingConfig # Assuming structure
# from pydantic import ValidationError

# --- Fixtures ---

# --- Remove config fixtures ---
# @pytest.fixture
# def mock_config_data():
#     """Provides raw dictionary data for a valid config."""
#     # ... (removed) ...

# @pytest.fixture
# def mock_validated_config(mock_config_data):
#     """Provides a mocked ProcessConfig object."""
#     # ... (removed) ...

@pytest.fixture
def executor_instance(tmp_path):
    """Creates a PipelineExecutor instance with real paths for output and checkpoint."""
    base_output_dir = tmp_path / "test_output"
    checkpoint_file = tmp_path / "test_checkpoint.json"
    log_file_path = base_output_dir / "pipeline_executor.log"

    # Mock logging handlers and Path.mkdir called during __init__'s _setup_logging
    with patch('epibench.pipeline.pipeline_executor.logging.FileHandler') as mock_file_handler, \
         patch('epibench.pipeline.pipeline_executor.logging.StreamHandler') as mock_stream_handler, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('epibench.pipeline.pipeline_executor.logger') as mock_init_logger: # Mock logger used during init

        # Instantiate with real Path objects for output dir
        executor = PipelineExecutor(base_output_directory=base_output_dir, checkpoint_file=checkpoint_file)
        
        # Assertions for initialization side effects
        mock_init_logger.info.assert_any_call(f"PipelineExecutor initialized. Output Dir: {base_output_dir}, Checkpoint: {checkpoint_file}")
        mock_mkdir.assert_any_call(parents=True, exist_ok=True) # Called for base_output_dir
        # Check if log file handler was set up (assuming INFO level)
        mock_file_handler.assert_called_once_with(log_file_path, mode='a')
        mock_stream_handler.assert_called_once_with(sys.stdout) # Check stream handler setup

        return executor

# --- Test Cases ---

def test_init_success_no_checkpoint(tmp_path):
    """Test successful initialization when checkpoint file doesn't exist."""
    base_output_dir = tmp_path / "output"
    checkpoint_file = tmp_path / "ckpt.json"
    log_file_path = base_output_dir / "pipeline_executor.log"

    # Mock logging and mkdir as in the fixture
    with patch('epibench.pipeline.pipeline_executor.logging.FileHandler') as mock_fh, \
         patch('epibench.pipeline.pipeline_executor.logging.StreamHandler') as mock_sh, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('epibench.pipeline.pipeline_executor.logger') as mock_logger:
         
        executor = PipelineExecutor(base_output_directory=base_output_dir, checkpoint_file=checkpoint_file)

    # --- Updated Assertions ---
    # assert executor.config == mock_validated_config # Removed
    assert executor.base_output_directory == base_output_dir
    assert executor.checkpoint_file == checkpoint_file
    assert executor.checkpoint_data == {} # Should be empty dict as file doesn't exist
    assert not checkpoint_file.exists() # Verify it wasn't created
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True) # Check mkdir called for base_output_dir
    mock_fh.assert_called_once_with(log_file_path, mode='a')
    mock_sh.assert_called_once_with(sys.stdout)
    mock_logger.info.assert_any_call(f"PipelineExecutor initialized. Output Dir: {base_output_dir}, Checkpoint: {checkpoint_file}")
    mock_logger.info.assert_any_call("Checkpoint file not found. Starting fresh.") # Check log message for checkpoint load

def test_init_success_with_checkpoint(tmp_path):
    """Test successful initialization loading existing checkpoint data."""
    base_output_dir = tmp_path / "output"
    checkpoint_file = tmp_path / "ckpt.json"
    checkpoint_content = {'sample1': {'status': 'completed'}}
    log_file_path = base_output_dir / "pipeline_executor.log"

    # Create the checkpoint file with content
    checkpoint_file.write_text(json.dumps(checkpoint_content))
    assert checkpoint_file.exists()

    # Mock logging and mkdir
    with patch('epibench.pipeline.pipeline_executor.logging.FileHandler') as mock_fh, \
         patch('epibench.pipeline.pipeline_executor.logging.StreamHandler') as mock_sh, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('epibench.pipeline.pipeline_executor.logger') as mock_logger:
         
        executor = PipelineExecutor(base_output_directory=base_output_dir, checkpoint_file=checkpoint_file)

    # --- Updated Assertions ---
    # assert executor.config == mock_validated_config # Removed
    assert executor.base_output_directory == base_output_dir
    assert executor.checkpoint_file == checkpoint_file
    assert executor.checkpoint_data == checkpoint_content # Should load content
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_fh.assert_called_once_with(log_file_path, mode='a')
    mock_sh.assert_called_once_with(sys.stdout)
    mock_logger.info.assert_any_call(f"PipelineExecutor initialized. Output Dir: {base_output_dir}, Checkpoint: {checkpoint_file}")
    mock_logger.info.assert_any_call(f"Loaded checkpoint data from {checkpoint_file}") # Check log message for checkpoint load

# --- Remove obsolete config validation test --- 
# def test_init_config_validation_error(tmp_path):
#     """Test initialization failure due to config validation error."""
#     # ... (removed) ...

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

def test_load_checkpoint_invalid_json(tmp_path):
    """Test _load_checkpoint with existing but malformed JSON file."""
    base_output_dir = tmp_path / "output"
    checkpoint_file = tmp_path / "ckpt.json" # Use a real path
    checkpoint_file.write_text("{malformed json") # Create the malformed file

    # --- Updated: Only mock logger and mkdir, no config validation patch needed ---
    with patch('epibench.pipeline.pipeline_executor.logger') as mock_logger, \
         patch('pathlib.Path.mkdir'): # Mock mkdir for logging setup

        executor = PipelineExecutor(base_output_directory=base_output_dir, checkpoint_file=checkpoint_file)

    # Check results after initialization
    assert executor.checkpoint_data == {}
    # --- Check exact error message might be brittle, check that *an* error was logged ---
    # Check that the specific error about checkpoint loading was logged *at some point*
    mock_logger.error.assert_any_call(f"Error loading checkpoint file {checkpoint_file}: Expecting property name enclosed in double quotes: line 1 column 2 (char 1). Starting fresh.")

def test_save_checkpoint(executor_instance):
    """Test _save_checkpoint writes correct data to the real path."""
    # --- This test should be mostly fine as executor_instance is updated ---
    save_path = executor_instance.checkpoint_file
    executor_instance.checkpoint_data = {'sampleA': {'status': 'failed'}, 'sampleB': {'status': 'completed'}}

    executor_instance._save_checkpoint()

    assert save_path.exists()
    with open(save_path, 'r') as f:
        saved_data = json.load(f)
    assert saved_data == executor_instance.checkpoint_data

# --- Tests for run() method ---

@pytest.fixture
def mock_run_dependencies(executor_instance):
    """Patch dependencies needed for the run() method test."""
    mock_subprocess_run = MagicMock(spec=subprocess.CompletedProcess)
    mock_subprocess_run.returncode = 0
    mock_subprocess_run.stdout = "Pipeline script finished."
    mock_subprocess_run.stderr = ""

    mock_temp_file_handle = mock_open().return_value 
    temp_file_name = "/tmp/dummy_temp_file.yaml"
    mock_temp_file_handle.name = temp_file_name
    mock_temp_file_obj = MagicMock()
    mock_temp_file_obj.__enter__.return_value = mock_temp_file_handle
    mock_temp_file_obj.__exit__.return_value = None

    # --- Update Path Mocking --- 
    # Get the real base_output_directory from the fixture
    base_output_dir_path = executor_instance.base_output_directory 
    script_path_str = "scripts/run_full_pipeline.py"

    mock_temp_path_instance = MagicMock(spec=Path, name="mock_temp_path")
    mock_temp_path_instance.exists.return_value = True
    mock_temp_path_instance.__str__.return_value = temp_file_name
    mock_temp_path_instance.unlink = MagicMock()

    mock_script_path_instance = MagicMock(spec=Path, name="mock_script_path")
    mock_script_path_instance.__str__.return_value = script_path_str

    # Mock the Path object created for the base output directory inside run()
    # This might not be strictly necessary if we just pass the string, but good practice
    mock_output_dir_instance = MagicMock(spec=Path, name="mock_output_dir")
    mock_output_dir_instance.__str__.return_value = str(base_output_dir_path)
    # Add mkdir mock on this specific instance for the results collector's potential use
    mock_output_dir_instance.mkdir = MagicMock()

    # --- Updated side effect for Path constructor --- 
    def run_path_side_effect(arg):
        str_arg = str(arg)
        if str_arg == script_path_str:
            return mock_script_path_instance
        # This check might be redundant if base_output_dir is passed directly as Path
        # elif str_arg == str(base_output_dir_path):
        #     return mock_output_dir_instance 
        elif str_arg == temp_file_name:
             return mock_temp_path_instance
        # Return a real Path object for other calls (like inside ResultsCollector)
        # Ensure the path is relative to the executor's base dir for consistency if needed
        # For simplicity, let's just return the real Path for now
        return Path(arg)

    # --- Mock ResultsCollector --- 
    mock_collector_instance = MagicMock(name="mock_collector_instance")
    mock_collector_instance.collect_all.return_value = {"summary": "mock results"} # Example return
    MockResultsCollector = MagicMock(name="MockResultsCollectorClass", return_value=mock_collector_instance)

    # Patch dependencies including the ResultsCollector
    with patch('epibench.pipeline.pipeline_executor.subprocess.run', return_value=mock_subprocess_run) as sp_run, \
         patch('epibench.pipeline.pipeline_executor.tempfile.NamedTemporaryFile', return_value=mock_temp_file_obj) as mock_tempfile_constructor, \
         patch('epibench.pipeline.pipeline_executor.Path', side_effect=run_path_side_effect) as mock_path_constructor_for_run, \
         patch('epibench.pipeline.pipeline_executor.yaml.dump') as mock_yaml_dump, \
         patch('epibench.pipeline.pipeline_executor.os.fsync'), \
         patch.object(executor_instance, '_save_checkpoint') as mock_save_ckpt, \
         patch('epibench.pipeline.pipeline_executor.logger') as mock_logger, \
         patch('epibench.pipeline.pipeline_executor.ResultsCollector', MockResultsCollector) as mock_collector_class: # Patch the class

        yield {
            "executor": executor_instance,
            "subprocess_run": sp_run,
            "tempfile_constructor": mock_tempfile_constructor,
            "temp_file_handle": mock_temp_file_handle,
            "yaml_dump": mock_yaml_dump,
            "temp_path_mock": mock_temp_path_instance,
            "save_checkpoint": mock_save_ckpt,
            "logger": mock_logger,
            "mock_path_constructor_for_run": mock_path_constructor_for_run,
            "mock_results_collector_class": mock_collector_class, # Yield the class mock
            "mock_results_collector_instance": mock_collector_instance # Yield the instance mock
        }

def test_run_all_samples_completed(mock_run_dependencies):
    """Test run() when all samples are already marked completed in checkpoint."""
    executor = mock_run_dependencies['executor']
    logger = mock_run_dependencies['logger']
    sp_run = mock_run_dependencies['subprocess_run']
    save_checkpoint = mock_run_dependencies['save_checkpoint']
    # --- Get collector mock --- 
    mock_collector_class = mock_run_dependencies['mock_results_collector_class']
    mock_collector_instance = mock_run_dependencies['mock_results_collector_instance']

    executor.checkpoint_data = {
        'sample1': {'status': 'completed'},
        'sample2': {'status': 'completed'}
    }
    sample_list = ['sample1', 'sample2']
    sample_details = {'sample1': {}, 'sample2': {}} 

    executor.run(sample_list, sample_details)

    logger.info.assert_any_call("Sample sample1 already completed. Skipping.")
    logger.info.assert_any_call("Sample sample2 already completed. Skipping.")
    logger.info.assert_any_call("No samples need processing in this run (all completed or skipped).")
    sp_run.assert_not_called()
    save_checkpoint.assert_not_called()
    # --- Assert collector was still called --- 
    mock_collector_class.assert_called_once_with(executor.base_output_directory, executor.checkpoint_data)
    mock_collector_instance.collect_all.assert_called_once()
    logger.info.assert_any_call("Pipeline execution run finished (no new samples processed).")

def test_run_some_samples_pending_success(mock_run_dependencies):
    """Test run() with pending/failed samples, successful subprocess execution."""
    executor = mock_run_dependencies['executor']
    sp_run = mock_run_dependencies['subprocess_run']
    yaml_dump = mock_run_dependencies['yaml_dump']
    save_checkpoint = mock_run_dependencies['save_checkpoint']
    temp_file_handle = mock_run_dependencies['temp_file_handle']
    temp_path_mock = mock_run_dependencies['temp_path_mock']
    mock_path_constructor_for_run = mock_run_dependencies['mock_path_constructor_for_run']
    # --- Get collector mock ---
    mock_collector_class = mock_run_dependencies['mock_results_collector_class']
    mock_collector_instance = mock_run_dependencies['mock_results_collector_instance']

    executor.checkpoint_data = {
        'sample1': {'status': 'completed'}, 
        'sample2': {'status': 'failed'},   
        'sample3': {}                      
    }
    sample_list = ['sample1', 'sample2', 'sample3', 'sample4'] 
    sample_details = {
        'sample1': {'process_data_config': 'cfg1'}, 
        'sample2': {'process_data_config': 'cfg2'},
        'sample3': {'process_data_config': 'cfg3'}
    }

    sp_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")

    executor.run(sample_list, sample_details)

    # --- Update Path constructor checks if needed (might be simpler now) ---
    # mock_path_constructor_for_run.assert_any_call("scripts/run_full_pipeline.py")
    # mock_path_constructor_for_run.assert_any_call(executor.base_output_directory)
    # mock_path_constructor_for_run.assert_any_call(temp_file_handle.name)

    sp_run.assert_called_once()
    args, kwargs = sp_run.call_args
    command_list = args[0]
    # --- Update command list assertions ---
    assert command_list[0] == sys.executable
    assert command_list[1] == "scripts/run_full_pipeline.py" # Uses __str__ from mock
    assert command_list[2] == "--samples-config"
    assert command_list[3] == temp_file_handle.name 
    assert command_list[4] == "--output-dir"
    # --- Use str() for Path comparison --- 
    assert command_list[5] == str(executor.base_output_directory) 
    assert command_list[6] == "--max-workers"
    assert command_list[7] == "1"

    yaml_dump.assert_called_once()
    dump_args, dump_kwargs = yaml_dump.call_args
    dumped_data = dump_args[0]
    assert len(dumped_data) == 2
    assert dumped_data[0]['name'] == 'sample2'
    assert dumped_data[1]['name'] == 'sample3'
    assert dump_args[1] == temp_file_handle

    save_checkpoint.assert_called_once()
    assert executor.checkpoint_data['sample1']['status'] == 'completed'
    assert executor.checkpoint_data['sample2']['status'] == 'completed'
    assert executor.checkpoint_data['sample3']['status'] == 'completed'
    assert 'sample4' not in executor.checkpoint_data

    temp_path_mock.unlink.assert_called_once_with()
    
    # --- Assert collector was called ---
    mock_collector_class.assert_called_once_with(executor.base_output_directory, executor.checkpoint_data)
    mock_collector_instance.collect_all.assert_called_once()

def test_run_subprocess_fails(mock_run_dependencies):
    """Test run() when the subprocess call fails."""
    executor = mock_run_dependencies['executor']
    sp_run = mock_run_dependencies['subprocess_run']
    yaml_dump = mock_run_dependencies['yaml_dump']
    save_checkpoint = mock_run_dependencies['save_checkpoint']
    logger = mock_run_dependencies['logger']
    temp_path_mock = mock_run_dependencies['temp_path_mock']
    # --- Get collector mock ---
    mock_collector_class = mock_run_dependencies['mock_results_collector_class']
    mock_collector_instance = mock_run_dependencies['mock_results_collector_instance']

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
    temp_path_mock.unlink.assert_called_once_with()
    # --- Assert collector was called ---
    mock_collector_class.assert_called_once_with(executor.base_output_directory, executor.checkpoint_data)
    mock_collector_instance.collect_all.assert_called_once()

# --- Remove obsolete config validation test --- 
# def test_run_aborts_if_config_invalid(tmp_path):
#     """Test that run() does not proceed if config failed validation."""
#     # ... (removed) ...

def test_run_temp_file_cleanup_on_error(mock_run_dependencies):
    """Test that temporary file is cleaned up even if subprocess fails."""
    executor = mock_run_dependencies['executor']
    sp_run = mock_run_dependencies['subprocess_run']
    temp_path_mock = mock_run_dependencies['temp_path_mock']
    # --- Get collector mock ---
    mock_collector_class = mock_run_dependencies['mock_results_collector_class']
    mock_collector_instance = mock_run_dependencies['mock_results_collector_instance']

    executor.checkpoint_data = {} 
    sample_list = ['sampleA']
    sample_details = {'sampleA': {'process_data_config': 'cfgA'}}
    
    sp_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="test cmd")

    executor.run(sample_list, sample_details)
    
    temp_path_mock.unlink.assert_called_once_with()
    # --- Assert collector was called ---
    mock_collector_class.assert_called_once_with(executor.base_output_directory, executor.checkpoint_data)
    mock_collector_instance.collect_all.assert_called_once()

# TODO: Add more tests:
# - Test logging setup variations (_setup_logging)
# - Test error during checkpoint saving/loading IOErrors
# - Test interaction with sample details more thoroughly (e.g., missing keys)
# - Test specific logging messages for different scenarios
# - Test _collect_results behavior when collector raises exception
