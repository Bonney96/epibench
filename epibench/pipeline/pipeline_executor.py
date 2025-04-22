import logging
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import tempfile
import os
import sys

# --- Remove unused imports ---
# from epibench.validation.config_validator import validate_process_config, ProcessConfig
# from pydantic import ValidationError
# --- Add Import ---
from epibench.pipeline.results_collector import ResultsCollector

logger = logging.getLogger(__name__) # Get logger instance

class PipelineExecutor:
    """
    Handles the systematic execution of the processing pipeline across multiple samples,
    including checkpointing and error handling. It is initialized with a base output
    directory and checkpoint file path.
    """

    def __init__(self, base_output_directory: Path, checkpoint_file: Path = Path('pipeline_checkpoint.json')):
        """
        Initializes the PipelineExecutor.

        Args:
            base_output_directory: The root directory where all outputs (per-sample subdirs, logs) will be stored.
            checkpoint_file: Path to the file used for saving and loading execution progress.
        """
        self.base_output_directory = Path(base_output_directory) # Ensure it's a Path object
        self.checkpoint_file = Path(checkpoint_file) # Ensure it's a Path object
        self.checkpoint_data = self._load_checkpoint()

        # --- Simplified Logging Setup ---
        self._setup_logging()
        logger.info(f"PipelineExecutor initialized. Output Dir: {self.base_output_directory}, Checkpoint: {self.checkpoint_file}")

    def _setup_logging(self):
        """Configures logging to console and a file within the base output directory."""
        # Ensure base output directory exists
        try:
            self.base_output_directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
             # Use print as logging might not be fully set up yet
             print(f"Error creating base output directory {self.base_output_directory}: {e}. Logging may be incomplete.", file=sys.stderr)
             # Continue logging setup, maybe just to console

        log_level = logging.INFO # Set a default level
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        log_file = self.base_output_directory / "pipeline_executor.log"

        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)

        # File Handler
        try:
            file_handler = logging.FileHandler(log_file, mode='a') # Append mode
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging initialized. Level: INFO, File: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file}: {e}. Logging to console only.")

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Loads the checkpoint data from the file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded checkpoint data from {self.checkpoint_file}")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading checkpoint file {self.checkpoint_file}: {e}. Starting fresh.")
                return {}
        else:
            logger.info("Checkpoint file not found. Starting fresh.")
            return {}

    def _save_checkpoint(self):
        """Saves the current checkpoint data to the file."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=4)
            logger.debug(f"Saved checkpoint data to {self.checkpoint_file}")
        except IOError as e:
            logger.error(f"Error saving checkpoint file {self.checkpoint_file}: {e}")

    def run(self, sample_list: List[str], sample_details: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Runs the full pipeline orchestration script (run_full_pipeline.py) for a batch of samples,
        respecting checkpoints. Generates a temporary sample config YAML for the script.

        Args:
            sample_list: A list of sample IDs to process.
            sample_details: Optional dictionary mapping sample IDs to their specific configurations
                            required by run_full_pipeline.py (e.g., 'process_data_config', 'train_config').
                            If None, assumes details are globally defined or not needed per sample.
        """
        if not sample_details:
             # If details aren't provided, create empty dicts.
             # run_full_pipeline.py might fail if required keys like
             # 'process_data_config' or 'train_config' are missing.
             # Consider adding a check or relying on run_full_pipeline.py validation.
             logger.warning("sample_details not provided to run(). Assuming defaults or global configs.")
             sample_details = {sample_id: {} for sample_id in sample_list}

        logger.info(f"Preparing pipeline run for {len(sample_list)} potential samples in {self.base_output_directory}.")

        # --- Filter Samples Based on Checkpoint --- 
        samples_to_process_this_run = []
        skipped_count = 0
        pending_or_failed_ids = [] 

        for sample_id in sample_list:
            if sample_id not in sample_details:
                logger.warning(f"No details found for sample {sample_id}. Skipping.")
                skipped_count += 1
                continue

            sample_status = self.checkpoint_data.get(sample_id, {}).get('status')
            if sample_status == 'completed':
                logger.info(f"Sample {sample_id} already completed. Skipping.")
                skipped_count += 1
                continue
            elif sample_status == 'failed':
                logger.warning(f"Sample {sample_id} failed previously. Retrying.")
                # Include failed samples in the list to be processed
            else: # No status or other status means pending
                 logger.info(f"Sample {sample_id} is pending.")

            # Construct the sample dict for the YAML file
            # Ensure 'name' key is present, defaulting to sample_id
            sample_config_for_yaml = sample_details[sample_id].copy()
            sample_config_for_yaml['name'] = sample_config_for_yaml.get('name', sample_id)
            samples_to_process_this_run.append(sample_config_for_yaml)
            pending_or_failed_ids.append(sample_id) # Track IDs being processed now

        if not samples_to_process_this_run:
            logger.info("No samples need processing in this run (all completed or skipped).")
            # --- Call results collector even if no samples processed in *this* run ---
            # This ensures a summary is always generated based on the checkpoint state
            self._collect_results()
            logger.info("Pipeline execution run finished (no new samples processed).")
            return

        logger.info(f"Will process {len(samples_to_process_this_run)} samples in this batch.")

        # --- Prepare and Run the Subprocess --- 
        success = False
        # Use a temporary file for the samples config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_yaml:
            temp_yaml_path = Path(tmp_yaml.name)
            try:
                yaml.dump(samples_to_process_this_run, tmp_yaml, default_flow_style=False)
                logger.info(f"Generated temporary samples config: {temp_yaml_path}")
                
                # Ensure temp file is flushed and closed before subprocess reads it
                tmp_yaml.flush()
                os.fsync(tmp_yaml.fileno())
                tmp_yaml.close() # Close here, subprocess needs to read it

                # --- Use self.base_output_directory ---
                # base_output_dir = self.config.output_directory
                # if not base_output_dir:
                #      logger.error("Base output directory not defined in the main configuration.")
                #      return # Cannot proceed without output dir
                base_output_dir = self.base_output_directory # Use the stored path

                command = [
                    sys.executable, # Use the same python interpreter
                    str(Path("scripts/run_full_pipeline.py")), # Ensure path is correct
                    '--samples-config', str(temp_yaml_path),
                    '--output-dir', str(base_output_dir), # Pass the correct output dir
                    '--max-workers', '1' # Run sequentially within the script as executor manages batch
                     # Add other necessary global arguments if run_full_pipeline.py supports them
                ]
                logger.info(f"Executing pipeline script: {' '.join(command)}")

                # Execute the main pipeline script ONCE for the batch
                result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')

                logger.info(f"Pipeline script completed successfully for the batch.")
                logger.debug(f"Subprocess stdout:\n{result.stdout}")
                if result.stderr:
                    logger.warning(f"Subprocess stderr:\n{result.stderr}")
                success = True

            except FileNotFoundError:
                logger.error(f"Error: The script 'scripts/run_full_pipeline.py' was not found.")
                success = False
            except subprocess.CalledProcessError as e:
                logger.error(f"Pipeline script failed for the batch with exit code {e.returncode}")
                logger.error(f"Stderr:\n{e.stderr}")
                logger.error(f"Stdout:\n{e.stdout}")
                success = False
            except Exception as e:
                logger.exception(f"An unexpected error occurred while executing the pipeline script: {e}")
                success = False
            finally:
                # Clean up the temporary file
                if temp_yaml_path.exists():
                    try:
                        temp_yaml_path.unlink()
                        logger.info(f"Removed temporary samples config: {temp_yaml_path}")
                    except OSError as e:
                        logger.error(f"Error removing temporary file {temp_yaml_path}: {e}")

        # --- Update Checkpoints for the Batch --- 
        final_status = 'completed' if success else 'failed'
        logger.info(f"Updating checkpoints for {len(pending_or_failed_ids)} processed/attempted samples with status: {final_status}")
        for sample_id in pending_or_failed_ids:
            if sample_id not in self.checkpoint_data:
                self.checkpoint_data[sample_id] = {}
            self.checkpoint_data[sample_id]['status'] = final_status
            # Optionally add a timestamp or error details here
        self._save_checkpoint()

        total_processed = len(pending_or_failed_ids)
        failed_count = 0 if success else total_processed
        logger.info("Pipeline batch run finished.")
        logger.info(f"Summary for this run: Samples Processed={total_processed}, Failed={failed_count}, Skipped Previously={skipped_count}")

        # --- Collect Results After Pipeline Run Attempt ---
        self._collect_results() # Call helper method

        logger.info("Pipeline execution run finished.")

    def _collect_results(self):
        """Helper method to instantiate and run the ResultsCollector."""
        logger.info("Attempting to collect results for all samples...")
        try:
            # --- Pass self.base_output_directory to ResultsCollector ---
            collector = ResultsCollector(self.base_output_directory, self.checkpoint_data)
            all_results = collector.collect_all() # collect_all likely returns None now or a summary dict

            # --- Adjust saving logic ---
            # The collector now handles saving reports within its own structure.
            # We might not need to save pipeline_summary.json here anymore,
            # or we could save a minimal summary returned by collect_all if needed.
            # For now, assume collector saves its reports and log completion.
            if all_results: # If collector returns something, maybe save it
                summary_path = self.base_output_directory / "pipeline_run_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(all_results, f, indent=4)
                logger.info(f"Pipeline run summary (from collector) saved to {summary_path}")
            else:
                logger.info("Results collection process completed (reports saved by collector).")

        except Exception as e:
            logger.exception(f"An error occurred during results collection: {e}")

# Example Usage (adjust as needed)
if __name__ == "__main__":
    # This is placeholder example usage.
    # The calling script/user is responsible for providing the configuration path
    # and determining the list of samples to process.

    # --- Argument Parsing ---
    import argparse
    parser = argparse.ArgumentParser(description="Run the EpiBench pipeline for multiple samples.")
    # --- Change --config to --output-dir ---
    parser.add_argument("-o", "--output-dir", required=True, type=Path,
                        help="Path to the base directory for all outputs.")
    parser.add_argument("-s", "--sample-list", required=True, type=Path,
                        help="Path to a file containing sample IDs (one per line).")
    # TODO: Add argument for sample details/metadata if needed, e.g., a sample sheet path.
    # Sample details should include paths to process/train configs per sample
    parser.add_argument("--sample-sheet", type=Path,
                         help="Path to a YAML/CSV file defining details (like config paths) per sample.")
    parser.add_argument("--checkpoint", type=Path, default=None, # Default to None, construct path later
                        help="Path to the checkpoint file (default: <output-dir>/pipeline_checkpoint.json).")

    args = parser.parse_args()

    # --- Validate Output Directory ---
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {args.output_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Set default checkpoint path if needed ---
    if args.checkpoint is None:
        args.checkpoint = args.output_dir / "pipeline_checkpoint.json"
    else:
        # Ensure parent dir exists if an explicit path is given
        try:
             args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
             print(f"Error creating directory for checkpoint file {args.checkpoint}: {e}", file=sys.stderr)
             sys.exit(1)

    # --- Sample List Loading ---
    if not args.sample_list.is_file():
        print(f"Error: Sample list file not found at {args.sample_list}")
        sys.exit(1)

    try:
        with open(args.sample_list, 'r') as f:
            samples_to_process = [line.strip() for line in f if line.strip()]
        if not samples_to_process:
             print(f"Error: Sample list file {args.sample_list} is empty.")
             sys.exit(1)
        print(f"Loaded {len(samples_to_process)} samples from {args.sample_list}")
    except Exception as e:
        print(f"Error reading sample list file {args.sample_list}: {e}")
        sys.exit(1)

    # --- Sample Details Loading (Placeholder) ---
    # TODO: Implement actual loading of sample details (process_data_config, train_config, etc.)
    #       required by run_full_pipeline.py from the --sample-sheet file.
    # This currently provides an empty dictionary for each sample, which will likely cause errors
    # in run_full_pipeline.py unless defaults are handled there.
    details_for_samples = {sample_id: {} for sample_id in samples_to_process}
    # Example structure needed in details_for_samples[sample_id]:
    # {
    #    'process_data_config': 'path/to/process_sample1.yaml',
    #    'train_config': 'path/to/train_sample1.yaml',
    #    'input_data_for_prediction': 'path/to/predict_input_sample1.h5' # Optional
    # }
    # --- Executor Initialization and Run ---
    try:
        # --- Update Executor Instantiation ---
        executor = PipelineExecutor(base_output_directory=args.output_dir, checkpoint_file=args.checkpoint)

        # --- Remove check for executor.config ---
        executor.run(sample_list=samples_to_process, sample_details=details_for_samples)

    except Exception as e:
         # Catch-all for unexpected errors during execution
         # Use print initially as logger setup happens in Executor init
         print(f"An unexpected error occurred: {e}", file=sys.stderr)
         # Attempt to log if possible
         logging.getLogger(__name__).exception(f"An unexpected error occurred during pipeline execution: {e}")
         sys.exit(1)

    print("Pipeline execution finished.") 